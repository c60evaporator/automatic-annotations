"""Grounding DINO のスタブ.

GPU や重みが無い環境で UI とジョブの流れを確認するためのもの。
settings.USE_STUB_MODELS = True のときに使われる。
本物と同じ predict() のシグネチャを保つこと。
"""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.services.postprocess import same_class_nms

logger = get_logger(__name__)

DEFAULT_STUB_DELAY_SEC = 0.12


class GroundingDinoDetectorStub:
    def __init__(self, *_: Any, **__: Any) -> None:
        logger.warning("Grounding DINO はスタブで動作しています（推論は行いません）")

    def to(self, device: str):
        return self

    def predict(
        self,
        image_path: Path,
        group_name: str,
        labels: list[str],
        *,
        score_threshold: float = 0.3,
        nms_same_class_iou: float = 0.6,
        stub_delay_sec: float | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        delay = DEFAULT_STUB_DELAY_SEC if stub_delay_sec is None else stub_delay_sec
        time.sleep(delay)

        # 画像パスとグループ名から決まる擬似乱数にして、
        # 再実行しても同じ結果が出るようにする（UI の確認がしやすい）
        seed = int(hashlib.md5(f"{image_path}:{group_name}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        boxes: list[dict[str, Any]] = []
        for _i in range(rng.randint(0, 5)):
            label = rng.choice(labels)
            score = round(rng.uniform(0.15, 0.98), 3)
            if score < score_threshold:
                continue
            x = rng.randint(0, 1400)
            y = rng.randint(0, 700)
            boxes.append({
                "xmin": x, "ymin": y,
                "xmax": min(x + rng.randint(60, 220), 1600),
                "ymax": min(y + rng.randint(50, 200), 900),
                "label": label, "score": score, "group": group_name,
            })
        return same_class_nms(boxes, nms_same_class_iou)
