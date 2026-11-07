"""PartHOE のスタブ（UI とジョブの流れの確認用）.

クロップの画素から決まる角度を返す。再実行しても同じ結果になる。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.models_impl.part_hoe import BIN_DEG, NUM_BINS

logger = get_logger(__name__)


class PartHOEEstimatorStub:
    def __init__(self, *_: Any, **__: Any) -> None:
        logger.warning("PartHOE はスタブで動作しています（推論は行いません）")

    def to(self, device: str) -> "PartHOEEstimatorStub":
        return self

    def predict(self, crops: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        probs = np.zeros((len(crops), NUM_BINS), dtype=np.float32)
        for i, crop in enumerate(crops):
            probs[i, int(np.asarray(crop).sum()) % NUM_BINS] = 1.0
        return probs

    @staticmethod
    def to_degrees(probs: np.ndarray) -> np.ndarray:
        return probs.argmax(axis=1) * BIN_DEG
