"""SAM2 トラッカー（Transformers 版）.

呼び出し側（routers/instance_tracking.py）は
「1 プロンプト区間 × 1 カメラ」で 1 回 `propagate()` を呼ぶ。
区間内で一貫した仮 ID（local_id）を返せばよく、
シーン全体を通した track_id の付け替えは呼び出し側の責務。

モデルは Transformers の Sam2VideoModel（既定 facebook/sam2.1-hiera-large）。
重みは HuggingFace から自動ダウンロードされ、HF_HOME に保存される（手動配置は不要）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.core.logging import get_logger
from app.models_impl.sam2_masks import masks_to_instances

logger = get_logger(__name__)


class Sam2Tracker:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
    ) -> None:
        # 重い import はここで行う。モジュールのトップに置くと
        # サーバー起動時に torch / transformers が読み込まれ、
        # 1 つでも壊れているとサーバー自体が起動しなくなる
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.model_name = model_name
        self.device = device
        # CUDA では bfloat16 で推論する（公式実装の autocast(bfloat16) に相当）
        self.dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

        logger.info("loading SAM2: %s", model_name)
        self.model = Sam2VideoModel.from_pretrained(model_name).to(device, dtype=self.dtype).eval()
        self.processor = Sam2VideoProcessor.from_pretrained(model_name)
        logger.info("SAM2 ready on %s", device)

    def to(self, device: str):
        """models.py の解放処理から呼ばれる（VRAM を返すため）."""
        self.model.to(device)
        self.device = device
        return self

    def propagate(
        self,
        frames: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
        *,
        dataroot: Path | str = "",
        mask_score_threshold: float = 0.5,
        direction: str = "forward",
        **_: Any,
    ) -> list[list[dict[str, Any]]]:
        """1 区間ぶんの伝播を行う.

        Args:
            frames: 区間内のフレーム（時刻順、sweep 含む）
            prompts: プロンプトのボックス
            direction: "forward" なら先頭フレームにプロンプトを与えて
                時間順に伝播。"backward" なら **最終フレーム**に与えて
                さかのぼる（区間途中で現れたインスタンスを拾うため）

        Returns:
            frames と同じ長さのリスト。各要素はそのフレームの
            インスタンス一覧（区間内で一貫した local_id 付き）。
        """
        from app.models_impl.sam2_video import (
            add_box_prompts,
            init_frame_state,
            propagate_inference,
        )
        import torch

        if not frames or not prompts:
            return [[] for _ in frames]

        root = Path(dataroot)
        images = [
            Image.open(root / frame["filename"]).convert("RGB") for frame in frames
        ]

        # 区間内の仮 ID を振る。SAM2 の obj_id としてそのまま使う
        prompt_boxes = [
            {**prompt, "local_id": index} for index, prompt in enumerate(prompts)
        ]
        local_to_label = {b["local_id"]: b.get("label") for b in prompt_boxes}
        local_to_score = {b["local_id"]: b.get("score") for b in prompt_boxes}
        local_to_detection = {
            b["local_id"]: b.get("detection_2d_id") for b in prompt_boxes
        }

        # プロンプトを与えるフレーム。backward は最終フレーム
        reverse = direction == "backward"
        prompt_frame_idx = len(frames) - 1 if reverse else 0

        # 推論セッションは区間ごとに新しく作るので、前の区間の状態は残らない
        with torch.inference_mode():
            inference_state = init_frame_state(
                self.model, self.processor, images, device=self.device, dtype=self.dtype
            )
            add_box_prompts(
                self.model, self.processor, inference_state, frame_idx=prompt_frame_idx,
                box_prompts=prompt_boxes,
            )
            propagated = propagate_inference(
                self.model, self.processor, inference_state,
                start_frame_idx=prompt_frame_idx, reverse=reverse,
            )

        height = int(inference_state.video_height)
        width = int(inference_state.video_width)
        # セッションはフレームと特徴量を GPU に保持しているので、ここで手放す
        del inference_state

        results: list[list[dict[str, Any]]] = []
        for frame_index in range(len(frames)):
            masks, obj_ids = propagated.get(
                frame_index, (None, [])
            )
            if masks is None or len(obj_ids) == 0:
                results.append([])
                continue

            instances = masks_to_instances(
                masks,
                image_height=height,
                image_width=width,
                local_ids=obj_ids,
                labels=[local_to_label.get(int(i)) for i in obj_ids],
                scores=[local_to_score.get(int(i)) for i in obj_ids],
                detection_ids=[local_to_detection.get(int(i)) for i in obj_ids],
                is_prompt_frame=(frame_index == prompt_frame_idx),
                threshold=0.5,  # マスクは既に logits>0 で二値化済み
            )
            results.append(instances)

        return results
