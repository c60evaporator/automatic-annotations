from transformers import Siglip2Model, Siglip2Processor
from PIL import Image
import torch

def batch_zero_shot_classify(
    model: Siglip2Model,
    processor: Siglip2Processor,
    images: list[Image.Image],
    candidate_labels: list[str],
    score_threshold: float = None,
) -> list[str]:
    """Batch zero-shot classification using SigLIP2

    Args:
        model: SigLIP2 model.
        processor: SigLIP2 processor.
        images: List of input images as PIL Images.
        candidate_labels: List of labels to detect in the images.
        score_threshold: Score threshold for filtering predictions.

    Returns:
        A list of predicted labels that meet the score threshold for each image.
    """
    # Create text prompts for each candidate label.
    texts = [f'This is a photo of {label}.' for label in candidate_labels]

    # Preprocess the inputs using the processor.
    inputs = processor(
        text=texts,
        images=images,
        padding="max_length",
        max_length=64,
        truncation=True,
        max_num_patches=256, 
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    sigmoid_scores = torch.sigmoid(logits_per_image)
    
    predicted_label_indices = torch.argmax(sigmoid_scores, dim=1)

    if score_threshold is not None:
        predicted_labels = [
            candidate_labels[i] for i, score in enumerate(sigmoid_scores[range(len(images)), predicted_label_indices])
            if score >= score_threshold
        ]
    else:
        predicted_labels = [candidate_labels[i] for i in predicted_label_indices]

    return predicted_labels

def zero_shot_classify(
    model: Siglip2Model,
    processor: Siglip2Processor,
    image: Image.Image,
    candidate_labels: list[str],
    score_threshold: float = None,
) -> str:
    """Zero-shot classification using SigLIP2

    Args:
        model: SigLIP2 model.
        image: Input image as a PIL Image.
        candidate_labels: List of labels to detect in the image.
        score_threshold: Score threshold for filtering predictions.
        device: Device to run the model on (e.g., "cuda" or "cpu").

    Returns:
        A predicted label that meet the score threshold.
    """
    return batch_zero_shot_classify(
        model=model,
        processor=processor,
        images=[image],
        candidate_labels=candidate_labels,
        score_threshold=score_threshold,
    )[0]
    

    


