"""Instance Tracking (SAM2) のエンドポイント.

現時点はプレースホルダ。2D 検出の挙動を詰めるあいだ、
サーバーが起動して /docs に構成が見えることだけを目的にしている。

実装するときは det2d.py と同じ形（POST でジョブ登録 → GET でポーリング）
に揃えること。3ステップで扱いが揃っていると UI 側を共通化できる。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/instance-tracking", tags=["Instance Tracking"])

KIND = "instance_tracking"


@router.get("/status")
def status() -> dict[str, str]:
    """このステップの実装状況を返す（起動確認用）."""
    return {"kind": KIND, "implemented": "false"}


@router.post("/jobs", status_code=501)
def create_tracking_job() -> None:
    """未実装.

    501 を返すのは、UI 側が「まだ動かない」ことを
    404（パス間違い）と区別できるようにするため。
    """
    raise HTTPException(
        status_code=501,
        detail="Instance Tracking は未実装です（2D Object Detection を先に調整中）",
    )
