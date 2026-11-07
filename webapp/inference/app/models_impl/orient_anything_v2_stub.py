"""Orient-Anything-V2 のスタブ（UI とジョブの流れの確認用）.

クロップの画素から決まる方位角を返す。再実行しても同じ結果になる。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class OrientAnythingV2EstimatorStub:
    def __init__(self, *_: Any, **__: Any) -> None:
        logger.warning("Orient-Anything-V2 はスタブで動作しています（推論は行いません）")

    def to(self, device: str) -> "OrientAnythingV2EstimatorStub":
        return self

    def predict(self, crops: list[np.ndarray], batch_size: int = 8) -> dict[str, np.ndarray]:
        n = len(crops)
        azimuth = np.array([int(np.asarray(c).sum()) % 360 for c in crops], dtype=np.float32)
        return {
            "azimuth_deg": azimuth,
            "elevation_deg": np.zeros(n, dtype=np.float32),
            "rotation_deg": np.zeros(n, dtype=np.float32),
            "alpha": np.ones(n, dtype=np.float32),
            "confidence": np.ones(n, dtype=np.float32),
        }
