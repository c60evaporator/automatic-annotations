import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import predict, annotate
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO

from .common.schemas import Box2D

def build_grounding_dino_prompt(labels: list[str]) -> str:
    """Create a prompt for Grounding DINO from a list of labels"""
    normalized = [
        label.lower().strip().rstrip(".")
        for label in labels
        if label.strip()
    ]
    if not normalized:
        raise ValueError("At least one label is required.")
    return ". ".join(normalized) + "."

def predict_multi_labels(
    model: GroundingDINO,
    image: torch.Tensor,
    labels: list[str],
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    ) -> list[Box2D]:
    """Run Grounding DINO prediction for a list of labels

    Args:
        model: Grounding DINO model.
        image: Input image as a torch tensor preprocessed by `groundingdino.util.inference.load_image`.
        labels: List of labels to detect in the image.
        box_threshold: Box threshold for filtering predictions.
        text_threshold: Text threshold for filtering predictions.

    Returns:
        A list of predicted Box2D objects. The box coordinates are in the format (x_min, y_min, x_max, y_max) in normalized coordinates (0 to 1).
    """
    prompt = build_grounding_dino_prompt(labels)
    boxes, logits, phrases = predict(
        model, image, prompt, box_threshold=box_threshold, text_threshold=text_threshold
    )
    xyxy_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    predicted_boxes = [
        Box2D(
            xyxy=box.cpu().numpy(),
            label=phrase,
            score=float(logit.cpu().numpy()),
        )
        for box, logit, phrase in zip(xyxy_boxes, logits, phrases)
    ]
    return predicted_boxes
