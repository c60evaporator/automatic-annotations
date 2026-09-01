"""Grounding DINO のラッパー（ルーターから使う入口）.

公式リポジトリ版（IDEA-Research/GroundingDINO）を使う。
重みは checkpoints ボリューム（既定 /opt/checkpoints）に手動で置く。

  GroundingDINO_SwinB_cfg.py : /opt/third_party/GroundingDINO/groundingdino/config/
  groundingdino_swinb_cogcoor.pth : /opt/checkpoints/

呼び出し側（routers/det2d.py）は「1フレーム × 1カテゴリグループ」で
1回 predict() を呼ぶ。スコア閾値と同一クラス NMS はグループ単位なので
この中で完結させる。クラスをまたぐ NMS は複数グループの結果が
揃わないと適用できないため、呼び出し側（フレーム単位）の責務。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.core.logging import get_logger
from app.models_impl.prompt import denormalize_boxes

logger = get_logger(__name__)


class GroundingDinoDetector:
    def __init__(
        self,
        config_path: str | Path,
        weight_path: str | Path,
        device: str = "cuda",
    ) -> None:
        # 重い import はここで行う。モジュールのトップに置くと、
        # サーバー起動時に torch と CUDA 拡張が読み込まれ、
        # 1つでも壊れているとサーバー自体が起動しなくなる
        from groundingdino.util.inference import load_model

        self.config_path = str(config_path)
        self.weight_path = str(weight_path)
        self.device = device

        if not Path(self.weight_path).exists():
            raise FileNotFoundError(
                f"Grounding DINO の重みが見つかりません: {self.weight_path}\n"
                "checkpoints フォルダに groundingdino_swinb_cogcoor.pth を "
                "配置してマウントしてください。"
            )
        if not Path(self.config_path).exists():
            raise FileNotFoundError(
                f"Grounding DINO の config が見つかりません: {self.config_path}"
            )

        logger.info("loading Grounding DINO: %s", self.weight_path)
        self.model = load_model(self.config_path, self.weight_path, device=device)
        logger.info("Grounding DINO ready on %s", device)

    def to(self, device: str):
        """models.py の解放処理から呼ばれる（VRAM を返すため）."""
        self.model.to(device)
        self.device = device
        return self

    def predict(
        self,
        image_path: Path,
        group_name: str,
        labels: list[str],
        *,
        score_threshold: float = 0.3,
        nms_same_class_iou: float = 0.6,
        nms_cross_class_iou: float = 0.85,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """1画像 × 1カテゴリグループの検出を行う.

        Returns:
            [{"xmin":.., "ymin":.., "xmax":.., "ymax":.., "label":.., "score":..}, ...]
            座標は画素単位の整数。
        """
        from app.models_impl.groundingdino_predict import predict_multi_labels

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        boxes, caption = predict_multi_labels(
            model=self.model,
            image=image,
            labels=labels,
            box_threshold=score_threshold,
            same_class_nms_iou=nms_same_class_iou,
            cross_class_nms_iou=nms_cross_class_iou,
            device=self.device,
        )
        logger.debug("caption=%r -> %d boxes", caption, len(boxes))

        # Grounding DINO は正規化座標を返すので、画像サイズを掛けて整数化する
        pixel_boxes = denormalize_boxes(
            [b["xyxy"] for b in boxes], image.width, image.height
        )
        return [
            {
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                "label": b["label"], "score": b["score"], "group": group_name,
            }
            for b, (xmin, ymin, xmax, ymax) in zip(boxes, pixel_boxes, strict=True)
        ]
