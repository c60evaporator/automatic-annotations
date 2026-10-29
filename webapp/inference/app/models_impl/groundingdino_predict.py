"""Grounding DINO（Transformers 版）の推論.

IDEA-Research/grounding-dino-base（公式 groundingdino_swinb_cogcoor の
Transformers 版）を使う。テキストプロンプトの扱いに結果が強く左右される
モデルなので、公式リポジトリ版で実験済みの手順（プロンプトの組み立て、
文字スパン→トークンの対応付け、ラベル単位の NMS）をそのまま踏襲している。

torch / transformers への import はこのモジュール内に閉じてある。
サーバー起動時ではなく、モデルの初回ロード時に読み込まれる
（app/core/models.py の遅延ローダー経由）。
"""
from __future__ import annotations

import torch
from PIL import Image
from torchvision.ops import batched_nms, box_convert
from transformers import (
    BatchEncoding,
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
)

from app.models_impl.prompt import PromptDefinition, build_multi_label_prompt


def preprocess_image(image: Image.Image, processor: GroundingDinoProcessor) -> torch.Tensor:
    """公式実装と同じ前処理（短辺 800 / 長辺 1333 以下へリサイズ + ImageNet 正規化）.

    Transformers の画像プロセッサの既定値が公式実装と同じ。
    Returns: (3, H, W) のテンソル
    """
    return processor.image_processor(images=image, return_tensors="pt")["pixel_values"][0]


def create_positive_map_from_span(
    tokenized: BatchEncoding,
    token_span: list[list[list[int]]],
    max_text_len: int = 256,
) -> torch.Tensor:
    """各ラベルの文字スパンを、対応するトークン位置のマップに変換する.

    groundingdino.util.vl_utils.create_positive_map_from_span の移植。
    fast tokenizer の char_to_token で文字位置→トークン位置を引く。

    Returns:
        (num_labels, max_text_len)。各行の合計が 1 になるよう正規化済み
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, label_spans in enumerate(token_span):
        for beg, end in label_spans:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None or end_pos is None:
                raise ValueError(f"文字スパン ({beg}, {end}) をトークンに対応付けられません")
            positive_map[j, beg_pos:end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def predict_multi_labels(
    model: GroundingDinoForObjectDetection,
    processor: GroundingDinoProcessor,
    image: Image.Image,
    labels: list[str],
    box_threshold: float = 0.3,
    same_class_nms_iou: float = 0.60,
    cross_class_nms_iou: float = 0.85,
    device: str = "cuda",
    sublabel_to_label: dict[str, str] | None = None,
) -> tuple[list[dict], str]:
    """Run Grounding DINO prediction for a list of labels

    Args:
        model: Grounding DINO model (Transformers).
        processor: Grounding DINO processor (Transformers).
        image: Input image as a PIL Image.
        labels: List of labels to detect in the image.
        box_threshold: Box threshold for filtering predictions.
        same_class_nms_iou: IoU threshold for non-maximum suppression (NMS)
            within the same class.
        cross_class_nms_iou: IoU threshold for NMS across different classes.
            Recommended to be higher than same_class_nms_iou to avoid removing
            boxes of different classes that overlap.
        device: Device to run the model on (e.g., "cuda" or "cpu").
        sublabel_to_label: サブラベル -> ラベルの逆引き。
            **same-class NMS はラベル単位で行う**ため、prompt に渡した語が
            サブラベル（car / van など）でも、同じラベルに畳み込まれる
            もの同士は統合される。

    Returns:
        (boxes, caption)
        boxes は {"xyxy": [...], "label": str, "sublabel": str, "score": float} のリスト。
        xyxy は 0〜1 に正規化された座標。画素座標への変換は呼び出し側で行う
        （画像サイズを知っているのは呼び出し側なので）。
    """
    # Build the prompt for multi-label detection
    prompt_definition: PromptDefinition = build_multi_label_prompt(labels)

    # Preprocess the image and the caption, and move them to the specified device
    image_tensor = preprocess_image(image, processor).to(device)
    tokenized = processor.tokenizer(prompt_definition.caption, return_tensors="pt")

    # Predict
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(
            pixel_values=image_tensor[None],
            input_ids=tokenized["input_ids"].to(device),
            token_type_ids=tokenized["token_type_ids"].to(device),
            attention_mask=tokenized["attention_mask"].to(device),
        )

    # shape: (num_queries, max_text_len)。プロンプト長より後ろは -inf（sigmoid で 0）
    token_scores = outputs.logits.float().cpu().sigmoid()[0]
    # shape: (num_queries, 4), normalized cxcywh
    boxes_cxcywh = outputs.pred_boxes.float().cpu()[0]

    # Create a matrix that maps each token to the corresponding label(s)
    # based on the token spans
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

    # Apply non-maximum suppression (NMS) within the same class.
    #
    # NOTE: idxs はサブラベルではなく**ラベル**の index にする。
    # サブラベルのままだと、同じ車が car と van で二重に残る
    mapping = sublabel_to_label or {}
    label_names = [
        mapping.get(name, name) for name in prompt_definition.raw_labels
    ]
    unique_labels = sorted(set(label_names))
    label_index_of = {name: i for i, name in enumerate(unique_labels)}
    sublabel_to_label_index = torch.tensor(
        [label_index_of[name] for name in label_names],
        dtype=torch.int64, device=kept_scores.device,
    )
    nms_class_indices = sublabel_to_label_index[kept_label_indices]

    keep_by_nms = batched_nms(
        boxes=boxes_xyxy,
        scores=kept_scores,
        idxs=nms_class_indices,
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
            # label は畳み込み後。sublabel はモデルが返した語そのもの
            "label": mapping.get(
                prompt_definition.raw_labels[int(label_index)],
                prompt_definition.raw_labels[int(label_index)],
            ),
            "sublabel": prompt_definition.raw_labels[int(label_index)],
            "score": float(score.item()),
        }
        for box, score, label_index in zip(
            boxes_xyxy, kept_scores, kept_label_indices, strict=True
        )
    ]
    return predicted_boxes, prompt_definition.caption
