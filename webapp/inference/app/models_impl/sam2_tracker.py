"""SAM2 トラッカー（公式リポジトリ版）.

呼び出し側（routers/instance_tracking.py）は
「1 プロンプト区間 × 1 カメラ」で 1 回 `propagate()` を呼ぶ。
区間内で一貫した仮 ID（local_id）を返せばよく、
シーン全体を通した track_id の付け替えは呼び出し側の責務。

重みは checkpoints ボリューム（既定 /opt/checkpoints）に手動で置く:
    sam2.1_hiera_large.pt
config は SAM2 パッケージ同梱の相対パスを Hydra が解決する:
    configs/sam2.1/sam2.1_hiera_l.yaml
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
        config_path: str,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        # 重い import はここで行う。モジュールのトップに置くと
        # サーバー起動時に torch と CUDA 拡張が読み込まれ、
        # 1 つでも壊れているとサーバー自体が起動しなくなる
        from sam2.build_sam import build_sam2_video_predictor

        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.device = device

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"SAM2 の重みが見つかりません: {self.checkpoint_path}\n"
                "checkpoints フォルダに sam2.1_hiera_large.pt を配置して"
                "マウントしてください。"
            )

        logger.info("loading SAM2: %s", self.checkpoint_path)
        # config はファイルパスではなく Hydra の設定名。
        # SAM2 パッケージ内の configs/ から解決される
        self.predictor = build_sam2_video_predictor(
            self.config_path, self.checkpoint_path, device=device
        )
        logger.info("SAM2 ready on %s", device)

    def to(self, device: str):
        """models.py の解放処理から呼ばれる（VRAM を返すため）."""
        self.predictor.to(device)
        self.device = device
        return self

    def propagate(
        self,
        frames: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
        *,
        dataroot: Path | str = "",
        mask_score_threshold: float = 0.5,
        **_: Any,
    ) -> list[list[dict[str, Any]]]:
        """1 区間ぶんの伝播を行う.

        Args:
            frames: 区間内のフレーム（時刻順、sweep 含む）。
                先頭がプロンプトを与えるキーフレーム
            prompts: 先頭フレームに与えるボックス

        Returns:
            frames と同じ長さのリスト。各要素はそのフレームの
            インスタンス一覧（区間内で一貫した local_id 付き）。
        """
        from app.models_impl.sam2_video import (
            _autocast,
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

        with torch.inference_mode(), _autocast(self.device):
            inference_state = init_frame_state(self.predictor, images)
            # 状態を作り直した直後でも、プロンプト追加前に reset しておく
            # （同じ predictor を区間ごとに使い回すため）
            self.predictor.reset_state(inference_state)
            add_box_prompts(
                self.predictor, inference_state, frame_idx=0,
                box_prompts=prompt_boxes,
            )
            propagated = propagate_inference(self.predictor, inference_state)

        height = inference_state["video_height"]
        width = inference_state["video_width"]

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
                is_prompt_frame=(frame_index == 0),
                threshold=0.5,  # マスクは既に logits>0 で二値化済み
            )
            results.append(instances)

        return results
