"""Grounding DINO のラッパー（ルーターから使う入口）.

Transformers 版（既定 IDEA-Research/grounding-dino-base。公式の
groundingdino_swinb_cogcoor に相当）を使う。重みは HuggingFace から
自動ダウンロードされ、HF_HOME（compose でホストの ~/.cache/huggingface を
マウント）に保存される。手動配置は不要。

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
        model_name: str,
        device: str = "cuda",
    ) -> None:
        # 重い import はここで行う。モジュールのトップに置くと、
        # サーバー起動時に torch / transformers が読み込まれ、
        # 1つでも壊れているとサーバー自体が起動しなくなる
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_name = model_name
        self.device = device

        logger.info("loading Grounding DINO: %s", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()
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
        sublabel_to_label: dict[str, str] | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """1画像 × 1カテゴリグループの検出を行う.

        Args:
            labels: プロンプトに渡す語（サブラベル）
            sublabel_to_label: サブラベル -> ラベルの逆引き。
                same-class NMS をラベル単位で行うために使う

        Returns:
            [{"xmin":.., .., "label":.., "sublabel":.., "score":..}, ...]
            座標は画素単位の整数。label は畳み込み後の値。
        """
        from app.models_impl.groundingdino_predict import predict_multi_labels

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        boxes, caption = predict_multi_labels(
            model=self.model,
            processor=self.processor,
            image=image,
            labels=labels,
            box_threshold=score_threshold,
            same_class_nms_iou=nms_same_class_iou,
            sublabel_to_label=sublabel_to_label,
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
                # label は畳み込み後、sublabel はモデルが返した語そのもの。
                # sublabel を落とすと UI の SubLabel 表示が空になる
                "label": b["label"], "sublabel": b.get("sublabel"),
                "score": b["score"], "group": group_name,
            }
            for b, (xmin, ymin, xmax, ymax) in zip(boxes, pixel_boxes, strict=True)
        ]
