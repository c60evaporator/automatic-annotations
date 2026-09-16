"""SigLIP2 のスタブ（UI とジョブの流れの確認用）.

入力から決まる乱数で候補のいずれかを返す。再実行しても同じ結果になるので、
「再判定でどのボックスが消えたか」を追いやすい。
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


class Siglip2ClassifierStub:
    def __init__(self, *_: Any, **__: Any) -> None:
        logger.warning("SigLIP2 はスタブで動作しています（推論は行いません）")

    def to(self, device: str):
        return self

    def classify_batch(
        self,
        images: Sequence[Image.Image],
        candidate_labels: Sequence[str],
        *,
        score_threshold: float | None = None,
    ) -> list[str | None]:
        if not images or not candidate_labels:
            return [None] * len(images)

        results: list[str | None] = []
        for image in images:
            # 画像サイズから決まる値で候補を選ぶ（決定的）
            seed = int(hashlib.md5(
                f"{image.size}:{len(candidate_labels)}".encode()
            ).hexdigest()[:8], 16)
            results.append(candidate_labels[seed % len(candidate_labels)])
        return results

    def classify(
        self,
        image: Image.Image,
        candidate_labels: Sequence[str],
        *,
        score_threshold: float | None = None,
    ) -> str | None:
        return self.classify_batch(
            [image], candidate_labels, score_threshold=score_threshold
        )[0]
