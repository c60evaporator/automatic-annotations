"""Grounding DINO（公式リポジトリ版）の推論.

HuggingFace 版ではなく IDEA-Research/GroundingDINO をそのまま使う。
テキストプロンプトの扱いに結果が強く左右されるモデルなので、
実験で動作確認済みの手順をそのまま踏襲している。

torch / groundingdino への import はこのモジュール内に閉じてある。
サーバー起動時ではなく、モデルの初回ロード時に読み込まれる
（app/core/models.py の遅延ローダー経由）。
"""
from __future__ import annotations

import torch
from PIL import Image
from torchvision.ops import batched_nms, box_convert

import groundingdino.datasets.transforms as T
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.vl_utils import create_positive_map_from_span

from app.models_impl.prompt import PromptDefinition, build_multi_label_prompt


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image, None)
    return image_transformed


def predict_multi_labels(
    model: GroundingDINO,
    image: Image.Image,
    labels: list[str],
    box_threshold: float = 0.3,
    same_class_nms_iou: float = 0.60,
    cross_class_nms_iou: float = 0.85,
    device: str = "cuda",
) -> tuple[list[dict], str]:
    """Run Grounding DINO prediction for a list of labels

    Args:
        model: Grounding DINO model.
        image: Input image as a PIL Image.
        labels: List of labels to detect in the image.
        box_threshold: Box threshold for filtering predictions.
        same_class_nms_iou: IoU threshold for non-maximum suppression (NMS)
            within the same class.
        cross_class_nms_iou: IoU threshold for NMS across different classes.
            Recommended to be higher than same_class_nms_iou to avoid removing
            boxes of different classes that overlap.
        device: Device to run the model on (e.g., "cuda" or "cpu").

    Returns:
        (boxes, caption)
        boxes は {"xyxy": [x1, y1, x2, y2], "label": str, "score": float} のリスト。
        xyxy は 0〜1 に正規化された座標。画素座標への変換は呼び出し側で行う
        （画像サイズを知っているのは呼び出し側なので）。
    """
    # Build the prompt for multi-label detection
    prompt_definition: PromptDefinition = build_multi_label_prompt(labels)

    # Preprocess the image and move it to the specified device
    image_tensor = preprocess_image(image).to(device)

    # Predict
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[prompt_definition.caption])

    # shape: (num_queries, max_text_len)
    token_scores = outputs["pred_logits"].cpu().sigmoid()[0]
    # shape: (num_queries, 4), normalized cxcywh
    boxes_cxcywh = outputs["pred_boxes"].cpu()[0]

    # Create a matrix that maps each token to the corresponding label(s)
    # based on the token spans
    tokenized = model.tokenizer(prompt_definition.caption)
    positive_map = create_positive_map_from_span(
        tokenized=tokenized,
        token_span=[
            [list(span) for span in label_spans]
            for label_spans in prompt_definition.character_spans
        ],
        max_text_len=token_scores.shape[1],
    ).to(
        device=token_scores.device,
        dtype=token_scores.dtype,
    )  # shape: (num_labels, max_text_len)
    # Calculate class scores by multiplying token scores with the positive map
    class_scores = token_scores @ positive_map.T

    # Get the label with the highest score for each box
    best_scores, best_label_indices = class_scores.max(dim=1)

    # Filter boxes based on the box threshold
    keep_by_score = best_scores > box_threshold
    kept_boxes = boxes_cxcywh[keep_by_score]
    kept_scores = best_scores[keep_by_score]
    kept_label_indices = best_label_indices[keep_by_score]

    # Convert the boxes to xyxy and filter out invalid boxes
    boxes_xyxy = box_convert(boxes=kept_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy = boxes_xyxy.clamp(0.0, 1.0)
    valid_geometry = (
        (boxes_xyxy[:, 2] > boxes_xyxy[:, 0])
        & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
    )
    boxes_xyxy = boxes_xyxy[valid_geometry]
    kept_scores = kept_scores[valid_geometry]
    kept_label_indices = kept_label_indices[valid_geometry]

    # Apply non-maximum suppression (NMS) within the same class
    keep_by_nms = batched_nms(
        boxes=boxes_xyxy,
        scores=kept_scores,
        idxs=kept_label_indices,
        iou_threshold=same_class_nms_iou,
    )
    boxes_xyxy = boxes_xyxy[keep_by_nms]
    kept_scores = kept_scores[keep_by_nms]
    kept_label_indices = kept_label_indices[keep_by_nms]

    # Apply non-maximum suppression (NMS) across different classes
    # Use a single group for cross-class NMS
    group_indices = torch.zeros(
        len(kept_scores), dtype=torch.int64, device=kept_scores.device
    )
    keep_cross_label = batched_nms(
        boxes=boxes_xyxy,
        scores=kept_scores,
        idxs=group_indices,
        iou_threshold=cross_class_nms_iou,
    )
    boxes_xyxy = boxes_xyxy[keep_cross_label]
    kept_scores = kept_scores[keep_cross_label]
    kept_label_indices = kept_label_indices[keep_cross_label]

    predicted_boxes = [
        {
            "xyxy": [float(v) for v in box.detach().cpu().tolist()],
            "label": prompt_definition.raw_labels[int(label_index)],
            "score": float(score.item()),
        }
        for box, score, label_index in zip(
            boxes_xyxy, kept_scores, kept_label_indices, strict=True
        )
    ]
    return predicted_boxes, prompt_definition.caption
