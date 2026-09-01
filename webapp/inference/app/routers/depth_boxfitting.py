"""Depth Estimation & Box Fitting (Depth-Anything-3) のエンドポイント.

現時点はプレースホルダ。instance_tracking.py と同じ扱い。
実装時は det2d.py のジョブ形式に揃える。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/depth-boxfitting", tags=["Depth Estimation & Box Fitting"])

KIND = "depth_boxfitting"


@router.get("/status")
def status() -> dict[str, str]:
    """このステップの実装状況を返す（起動確認用）."""
    return {"kind": KIND, "implemented": "false"}


@router.post("/jobs", status_code=501)
def create_depth_job() -> None:
    """未実装."""
    raise HTTPException(
        status_code=501,
        detail="Depth Estimation & Box Fitting は未実装です（2D Object Detection を先に調整中）",
    )
