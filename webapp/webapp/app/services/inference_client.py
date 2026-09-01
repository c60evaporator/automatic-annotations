"""推論サーバー（FastAPI）との通信.

ジョブ方式なので、クライアント側は
「投げる → 進捗を取りに行く → 必要ならキャンセル」だけを担う。
推論そのものはサーバー側で走り続けるため、Streamlit の再実行で
接続が切れても推論は失われない。
"""
from __future__ import annotations

from typing import Any

import requests

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 状態取得は軽いので短く。ジョブ登録も即座に返る（202）ので長くしない
POLL_TIMEOUT_SEC = 10.0
SUBMIT_TIMEOUT_SEC = 30.0

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class InferenceServerError(RuntimeError):
    """推論サーバーとの通信に失敗した."""


def _url(path: str) -> str:
    return f"{get_settings().INFERENCE_BASE_URL.rstrip('/')}{path}"


def _request(method: str, path: str, *, timeout: float, **kwargs: Any) -> dict[str, Any]:
    try:
        res = requests.request(method, _url(path), timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise InferenceServerError(f"推論サーバーに接続できません: {exc}") from exc

    if res.status_code >= 400:
        detail = ""
        try:
            detail = res.json().get("detail", "")
        except ValueError:
            detail = res.text[:200]
        raise InferenceServerError(f"HTTP {res.status_code}: {detail}")
    return res.json()


def health() -> dict[str, Any]:
    return _request("GET", "/health", timeout=POLL_TIMEOUT_SEC)


def is_available() -> bool:
    """推論サーバーが応答するか（UI の事前チェック用）."""
    try:
        health()
        return True
    except InferenceServerError:
        return False


def submit_detection2d(payload: dict[str, Any]) -> dict[str, Any]:
    """2D 検出ジョブを登録し、job 情報を返す."""
    return _request("POST", "/detection2d/jobs",
                    timeout=SUBMIT_TIMEOUT_SEC, json=payload)


def get_detection2d_job(job_id: str, since: int = 0) -> dict[str, Any]:
    """ジョブの進捗と、since 以降の部分結果を取得する.

    since に受け取り済みの件数を渡すことで、ポーリングのたびに
    全結果を送り直させない。
    """
    return _request("GET", f"/detection2d/jobs/{job_id}",
                    timeout=POLL_TIMEOUT_SEC, params={"since": since})


def cancel_detection2d_job(job_id: str) -> dict[str, Any]:
    return _request("DELETE", f"/detection2d/jobs/{job_id}",
                    timeout=POLL_TIMEOUT_SEC)
