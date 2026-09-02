"""SAM2 トラッカーのスタブ.

UI とジョブの流れを確認するためのもので、実際の推論は入っていない。
本実装に差し替えるときは `propagate()` のシグネチャを保つこと。

呼び出し側（routers/instance_tracking.py）は
「1 プロンプト区間 × 1 カメラ」で 1 回呼ぶ想定。
区間境界での track_id 引き継ぎは呼び出し側の責務なので、
ここでは区間内で一貫した仮 id（区間内 index）を返せばよい。
"""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from common.mask_rle import rle_area, rle_from_box

logger = get_logger(__name__)

DEFAULT_STUB_DELAY_SEC = 0.15

# スタブが返すマスクの既定サイズ（frame に width/height が無い場合）
DEFAULT_IMAGE_SIZE = (900, 1600)


class Sam2TrackerStub:
    def __init__(self, *_: Any, **__: Any) -> None:
        logger.warning("SAM2 はスタブで動作しています（推論は行いません）")

    def to(self, device: str):
        return self

    def propagate(
        self,
        frames: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
        *,
        dataroot: Path | str = "",
        mask_score_threshold: float = 0.5,
        stub_delay_sec: float | None = None,
        **_: Any,
    ) -> list[list[dict[str, Any]]]:
        """1 区間ぶんの伝播を行う.

        Args:
            frames: 区間内のフレーム（sample_idx 昇順、sweep 含む）。
                先頭がプロンプトを与える sample のキーフレーム。
            prompts: 先頭フレームに与えるボックス

        Returns:
            frames と同じ長さのリスト。各要素はそのフレームの
            インスタンス一覧（区間内で一貫した仮 track_id 付き）。
        """
        delay = DEFAULT_STUB_DELAY_SEC if stub_delay_sec is None else stub_delay_sec
        time.sleep(delay)

        if not frames:
            return []

        height = frames[0].get("height") or DEFAULT_IMAGE_SIZE[0]
        width = frames[0].get("width") or DEFAULT_IMAGE_SIZE[1]

        # フレームごとに少しずつボックスを動かして、伝播の見た目を作る。
        # 同じ入力なら同じ結果になるよう、パスから決まる乱数にする
        seed = int(hashlib.md5(
            f"{frames[0]['sample_data_token']}".encode()
        ).hexdigest()[:8], 16)
        rng = random.Random(seed)
        drifts = [
            (rng.randint(-8, 8), rng.randint(-5, 5)) for _ in range(len(prompts))
        ]

        results: list[list[dict[str, Any]]] = []
        for step, frame in enumerate(frames):
            instances: list[dict[str, Any]] = []
            for index, prompt in enumerate(prompts):
                dx, dy = drifts[index]
                xmin = max(0, min(width - 2, prompt["xmin"] + dx * step))
                ymin = max(0, min(height - 2, prompt["ymin"] + dy * step))
                xmax = max(xmin + 1, min(width, prompt["xmax"] + dx * step))
                ymax = max(ymin + 1, min(height, prompt["ymax"] + dy * step))

                mask_rle = rle_from_box(xmin, ymin, xmax, ymax, height, width)
                instances.append({
                    # 区間内で一貫していればよい仮 id。
                    # シーン全体の track_id は呼び出し側が付け替える
                    "local_id": index,
                    "label": prompt["label"],
                    "score": prompt.get("score"),
                    "mask_rle": mask_rle,
                    "mask_area": rle_area(mask_rle),
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                    "detection_2d_id": prompt.get("detection_2d_id"),
                    "is_prompt_frame": step == 0,
                })
            results.append(instances)
        return results
