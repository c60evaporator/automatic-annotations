"""Grounding DINO のラッパー.

現状は**スタブ**。指定秒だけ待って、それらしいボックスを返す。
UI とジョブの流れを詰めるためのもので、実際の推論は入っていない。

本実装に差し替えるときは predict() のシグネチャを保つこと。
呼び出し側（routers/det2d.py）は
「1フレーム × 1カテゴリグループ」で1回呼ぶ想定。

スコア閾値と同一クラス NMS はグループ単位なので、この中で完結させる。
クラスをまたぐ NMS は複数グループの結果が揃わないと適用できないため、
呼び出し側（フレーム単位）の責務にしてある。
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

# スタブが1推論あたり待つ既定秒数
DEFAULT_STUB_DELAY_SEC = 0.12


class GroundingDinoDetector:
    def __init__(self, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device = device
        # 本実装ではここでモデルをロードする
        logger.info("GroundingDinoDetector (stub) ready: %s on %s", model_id, device)

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
        """1画像 × 1カテゴリグループの検出を行う.

        Args:
            labels: このグループでプロンプトにするラベル一覧
            score_threshold: このグループのスコア閾値
            nms_same_class_iou: このグループ内の同一クラス NMS IoU

        Returns:
            [{"xmin":.., "ymin":.., "xmax":.., "ymax":.., "label":.., "score":..}, ...]
        """
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
            w = rng.randint(60, 220)
            h = rng.randint(50, 200)
            boxes.append({
                "xmin": x, "ymin": y,
                "xmax": min(x + w, 1600), "ymax": min(y + h, 900),
                "label": label, "score": score, "group": group_name,
            })

        return same_class_nms(boxes, nms_same_class_iou)
