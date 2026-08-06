from dataclasses import dataclass

import torch
from torchvision.ops import box_convert, batched_nms
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO

from ..common.schemas import Box2D


@dataclass(frozen=True)
class PromptDefinition:
    caption: str
    raw_labels: tuple[str, ...]
    character_spans: tuple[tuple[tuple[int, int], ...], ...]


def build_multi_label_prompt(labels: list[str]) -> str:
    """Create a prompt for Grounding DINO from a list of labels

    Args:
        labels: List of labels to detect in the image.
    
    Returns:
        A PromptDefinition object containing the caption, raw labels, and token spans.

        example:
        labels = ["cat", "dog", "bird"]
        returns:
        PromptDefinition(
            caption="cat. dog. bird.",
            raw_labels=("cat", "dog", "bird"),
            character_spans=(((0, 3),), ((5, 8),), ((10, 14),)),
        )
    """
    caption = ""
    raw_labels: list[str] = []
    character_spans: list[tuple[tuple[int, int], ...]] = []

    seen_labels: set[str] = set()
    prompt_to_canonical_label: dict[str, str] = {}

    for raw_label in labels:
        # lowercase, strip whitespace, and remove trailing periods
        canonical_label = raw_label.lower().strip().rstrip(".")
        
        # remove duplicate labels and empty labels
        if not canonical_label:
            continue
        if canonical_label in seen_labels:
            continue
        seen_labels.add(canonical_label)
        
        # replace underscores with spaces and normalize whitespace
        prompt_label = canonical_label.replace("_", " ")
        prompt_label = " ".join(prompt_label.split())

        # Check for conflicting labels that produce the same prompt phrase
        previous_label = prompt_to_canonical_label.get(prompt_label)
        if (
            previous_label is not None
            and previous_label != canonical_label
        ):
            raise ValueError(
                "Different labels produce the same prompt phrase: "
                f"{previous_label!r} and {canonical_label!r}"
            )
        prompt_to_canonical_label[prompt_label] = canonical_label

        # Add the space to the prompt if it's not the first label
        if caption:
            caption += " "

        # Store the start and end indices of the prompt label in the caption to restore the original labels later
        start = len(caption)
        caption += prompt_label
        end = len(caption)
        character_spans.append(((start, end),))

        # Store the original label for later use
        raw_labels.append(canonical_label)
        caption += "."

    if not raw_labels:
        raise ValueError("At least one label is required.")

    return PromptDefinition(
        caption=caption,
        raw_labels=tuple(raw_labels),
        character_spans=tuple(character_spans),
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image, None)
    return image_transformed


def _make_nms_group_indices(
    label_indices: torch.Tensor,
    raw_labels: tuple[str, ...],
    label_to_group: dict[str, str],
) -> torch.Tensor:
    """相互排他的なラベルを同じNMSグループへ割り当てる。"""
    group_to_index: dict[str, int] = {}
    result: list[int] = []

    for label_index in label_indices.tolist():
        label = raw_labels[label_index]

        # グループ未指定ラベルは、他ラベルと干渉しない独自グループにする
        group_name = label_to_group.get(
            label,
            f"__label__:{label}",
        )

        if group_name not in group_to_index:
            group_to_index[group_name] = len(group_to_index)

        result.append(group_to_index[group_name])

    return torch.tensor(
        result,
        dtype=torch.int64,
        device=label_indices.device,
    )


def predict_multi_labels(
    model: GroundingDINO,
    image: Image.Image,
    labels: list[str],
    box_threshold: float = 0.3,
    same_class_nms_iou: float = 0.60,
    cross_class_nms_iou: float = 0.85,
    device: str = "cuda",
    ) -> list[Box2D]:
    """Run Grounding DINO prediction for a list of labels

    Args:
        model: Grounding DINO model.
        image: Input image as a PIL Image.
        labels: List of labels to detect in the image.
        box_threshold: Box threshold for filtering predictions.
        same_class_nms_iou: IoU threshold for non-maximum suppression (NMS) within the same class.
        cross_class_nms_iou: IoU threshold for NMS across different classes. Recommended to be higher than same_class_nms_iou to avoid removing boxes of different classes that overlap.
        device: Device to run the model on (e.g., "cuda" or "cpu").

    Returns:
        A list of predicted Box2D objects. The box coordinates are in the format (x_min, y_min, x_max, y_max) in normalized coordinates (0 to 1).
    """
    # Build the prompt for multi-label detection
    prompt_definition = build_multi_label_prompt(labels)

    # Preprocess the image and move it to the specified device
    image = preprocess_image(image).to(device)

    # Predict
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image[None], captions=[prompt_definition.caption])

    token_scores = outputs["pred_logits"].cpu().sigmoid()[0]  # shape: (num_queries, max_text_len)
    boxes_cxcywh = outputs["pred_boxes"].cpu()[0]  # shape: (num_queries, 4), normalized cxcywh

    # Create a matrix that maps each token to the corresponding label(s) based on the token spans
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
    # Calculate class scores by multiplying token scores with the positive map matrix
    class_scores = token_scores @ positive_map.T

    # Get the label with the highest score for each box
    best_scores, best_label_indices = class_scores.max(dim=1)

    # Filter boxes based on the box threshold
    keep_by_score  = best_scores > box_threshold
    kept_boxes = boxes_cxcywh[keep_by_score]
    kept_scores = best_scores[keep_by_score]
    kept_label_indices = best_label_indices[keep_by_score]

    # Convert the boxes to xyxy and filter out invalid boxes (where x_min >= x_max or y_min >= y_max)
    boxes_xyxy = box_convert(boxes=kept_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy = boxes_xyxy.clamp(0.0, 1.0)  # Limit the box coordinates to be within [0, 1]
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
    group_indices = torch.zeros(len(kept_scores), dtype=torch.int64,  # Use a single group for cross-class NMS
                                device=kept_scores.device)
    keep_cross_label = batched_nms(
        boxes=boxes_xyxy,
        scores=kept_scores,
        idxs=group_indices,
        iou_threshold=cross_class_nms_iou,
    )
    boxes_xyxy = boxes_xyxy[keep_cross_label]
    kept_scores = kept_scores[keep_cross_label]
    kept_label_indices = kept_label_indices[keep_cross_label]

    # Store predicted boxes as Box2D objects
    predicted_boxes = [
        Box2D(
            xyxy=box.detach().numpy(),
            label=prompt_definition.raw_labels[label_index],
            score=float(score.item()),
        )
        for box, score, label_index in zip(boxes_xyxy, kept_scores, kept_label_indices, strict=True)
    ]
    return predicted_boxes, prompt_definition.caption
