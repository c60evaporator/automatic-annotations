"""非同期ジョブの管理.

シーン単位の推論は数分かかる。同期 API にすると:
  - HTTP のタイムアウトに引っかかる
  - Streamlit 側が固まって進捗を出せない
  - 途中でキャンセルできない

ため、「POST でジョブを登録して job_id を返す → GET でポーリング」
という形にする。Streamlit は st.rerun() でポーリングすればよい。

ジョブはプロセス内のメモリに保持する。推論サーバーは worker=1 の
単一プロセスで動かす前提なのでこれで足りる。
永続化が必要になったら DB か Redis に移す。
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    kind: str                       # 'detection_2d' | 'instance_tracking' | 'depth_boxfitting'
    status: JobStatus = JobStatus.PENDING
    total: int = 0                  # 処理対象数（推論の実行回数など）
    processed: int = 0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    # 完了した単位（フレーム等）を逐次ためる。
    # 完了を待たずに UI へ流すためのバッファで、最終的な result とは別物。
    partial: list[dict[str, Any]] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def progress(self) -> float:
        return self.processed / self.total if self.total else 0.0

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.ended_at or time.time()) - self.started_at

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def set_progress(self, processed: int, total: int | None = None,
                     message: str = "") -> None:
        self.processed = processed
        if total is not None:
            self.total = total
        if message:
            self.message = message

    def append_partial(self, item: dict[str, Any]) -> None:
        """完了した1単位を追加する（ポーリング側が差分で取りに来る）."""
        with self._lock:
            self.partial.append(item)

    def partial_since(self, since: int) -> list[dict[str, Any]]:
        """since 番目以降の部分結果を返す.

        毎回全件返すとポーリングのたびに同じデータを送ることになるため、
        クライアントが受け取り済みの件数を渡して差分だけ取る。
        """
        with self._lock:
            return self.partial[since:]

    def to_dict(self, since: int | None = None) -> dict[str, Any]:
        data = {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "total": self.total,
            "processed": self.processed,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "elapsed_sec": round(self.elapsed, 2),
            "result": self.result,
            "partial_count": len(self.partial),
        }
        if since is not None:
            data["partial"] = self.partial_since(since)
        return data


class JobManager:
    """ジョブの登録・実行・参照.

    ワーカーは1本だけ。GPU が1枚なので同時に走らせても得がなく、
    直列化しておくほうが VRAM の見積もりが立つ。
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="inference"
        )

    def submit(self, kind: str, fn: Callable[[Job], dict[str, Any]]) -> Job:
        """ジョブを登録する.

        fn は Job を受け取り、進捗を job.set_progress() で報告し、
        結果の dict を返す。job.cancel_requested() を定期的に確認して
        中断に応じること。
        """
        job = Job(id=str(uuid.uuid4()), kind=kind)
        with self._lock:
            self._jobs[job.id] = job
            self._futures[job.id] = self._pool.submit(self._run, job, fn)
        self._cleanup()
        return job

    def _run(self, job: Job, fn: Callable[[Job], dict[str, Any]]) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        try:
            result = fn(job)
            if job.cancel_requested():
                job.status = JobStatus.CANCELLED
                job.message = "キャンセルされました"
            else:
                job.result = result
                job.status = JobStatus.SUCCEEDED
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            logger.error("job %s failed\n%s", job.id, traceback.format_exc())
        finally:
            job.ended_at = time.time()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, kind: str | None = None) -> list[Job]:
        jobs = list(self._jobs.values())
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        """キャンセルを要求する.

        実行中のジョブは即座には止まらない。ジョブ関数が
        cancel_requested() を確認した時点で終了する。
        """
        job = self._jobs.get(job_id)
        if job is None or job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED,
                                         JobStatus.CANCELLED):
            return False
        job._cancel.set()
        future = self._futures.get(job_id)
        if future is not None and future.cancel():
            # まだ実行前ならその場で確定できる
            job.status = JobStatus.CANCELLED
            job.ended_at = time.time()
        return True

    def _cleanup(self) -> None:
        """古い完了ジョブを捨てる（メモリの上限を作る）."""
        retention = get_settings().JOB_RETENTION_SEC
        now = time.time()
        with self._lock:
            stale = [
                jid for jid, j in self._jobs.items()
                if j.ended_at is not None and now - j.ended_at > retention
            ]
            for jid in stale:
                self._jobs.pop(jid, None)
                self._futures.pop(jid, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """JobManager のシングルトン（FastAPI の依存性として使う）."""
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def reset_job_manager() -> None:
    """シングルトンを破棄する（lifespan の終了時に呼ぶ）.

    shutdown 済みの ThreadPoolExecutor は新しいジョブを受け付けず、
    "cannot schedule new futures after shutdown" になる。
    同一プロセスでアプリを作り直す場合（テスト等）に備えて捨てておく。
    """
    global _manager
    _manager = None
