from typing import Optional
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor

from ..common.schemas import Box2D, Instance2D
from ..common.geometry.crop_resize import resize_mask_nearest


def _masks_to_instances(
    masks: np.ndarray,
    image_height: int,
    image_width: int,
    labels: list[str] | None = None,
    scores: np.ndarray | None = None,
    boxes: np.ndarray | None = None,
    threshold: float = 0.5,
) -> list[Instance2D]:
    """
    Converts the predicted masks to a list of Instance2D objects.

    Args:
        masks: The predicted masks as a NumPy array of shape (C, H, W).
        image_height: The height of the image before resizing.
        image_width: The width of the image before resizing.
        labels: The labels for each instance.
        scores: The confidence scores for each instance.
        boxes: The bounding boxes for each instance.
        threshold: The threshold to binarize the masks.
    
    Returns:
        A list of Instance2D objects corresponding to the non-empty masks.
    """
    if masks.ndim != 3:
        raise ValueError(
            f"masks must have shape (C, H, W), but got {masks.shape}."
        )
    
    num_masks = masks.shape[0]
    if num_masks == 0:
        return []
    
    # Resize masks to the original image size if necessary.
    # bilinear interpolation before thresholding, the boundaries of the masks become smoother.
    if masks.shape[-2:] != (image_height, image_width):
        masks = resize_mask_nearest(masks, image_height, image_width)
    binary = masks > threshold  # (C, H, W) bool

    # Calculate bounding rectangle of the mask
    rows = binary.any(axis=2)  # (C, H) If a row has any True value, it is part of the mask
    cols = binary.any(axis=1)  # (C, W) If a column has any True value, it is part of the mask
    nonempty = rows.any(axis=1)  # (C,) If a mask has any True value, it is non-empty
    # argmax returns the index of the first True value.
    # Flipping and subtracting gives the "end + 1" (exclusive) index.
    y0 = rows.argmax(axis=1)
    y1 = rows.shape[1] - np.flip(rows, axis=1).argmax(axis=1)
    x0 = cols.argmax(axis=1)
    x1 = cols.shape[1] - np.flip(cols, axis=1).argmax(axis=1)

    # 3. Create Instance2D objects for each non-empty mask
    regions = np.stack([x0, y0, x1, y1], axis=1).astype(np.int64)

    instances: list[Instance2D] = []

    for index in range(num_masks):
        # Skip empty masks
        if not nonempty[index]:
            continue
    
        # Get the bounding box coordinates for the current mask
        rx0, ry0, rx1, ry1 = (int(value) for value in regions[index])
        # mask_region uses exclusive x1/y1 coordinates.
        mask = binary[index, ry0:ry1, rx0:rx1]
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
    
        if boxes is not None:
            xyxy = np.asarray(boxes[index], dtype=np.float64).copy()
        else:
            # Pixel i represents the continuous coordinate interval
            # [i, i + 1), so the exclusive rx1/ry1 values can be used
            # directly as the bottom-right coordinates.
            xyxy = np.array([rx0, ry0, rx1, ry1], dtype=np.float64)

        instances.append(
            Instance2D(
                box=Box2D(
                    xyxy=xyxy,
                    label=None if labels is None else labels[index],
                    score=None if scores is None else float(scores[index]),
                ),
                mask=mask,
                mask_region=(rx0, ry0, rx1, ry1),
            )
        )
    return instances

def predict(
    predictor: SAM2ImagePredictor,
    image: Image.Image,
    point_coords: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    box: Box2D | None = None,
    multimask_output: bool = False,
    device: str = "cuda",
) -> list[Instance2D]:
    """
    Predicts the segmentation mask for a given image using the provided model.

    Args:
        predictor: SAM2ImagePredictor instance used for making predictions.
        image: The input image for which to predict the segmentation mask.
        point_coords: The coordinates of the points for interactive segmentation.
        point_labels: The labels of the points for interactive segmentation.
        box: The bounding box for the object to segment. The coordinates should be **pixel values** in the xyxy format.
        multimask_output: Whether to output multiple masks per instance. Default is False.
        x_offset: The horizontal offset of the cropped image within the original image. Default is 0.
        y_offset: The vertical offset of the cropped image within the original image. Default is 0.
        device: The device to run the model on. Default is "cuda". Can be set to "cpu" for CPU inference.

    Returns:
        A list of Instance2D objects representing the predicted instances.
    """
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    # Inference
    with torch.inference_mode(), autocast_context:
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )

    # Sort masks by scores in descending order
    if multimask_output:
        sorted_indices = np.argsort(scores)[::-1]
        masks = masks[sorted_indices]
        scores = scores[sorted_indices]

    # Convert masks to Instance2D objects
    instances = _masks_to_instances(
        masks=masks,
        image_height=image.height,
        image_width=image.width,
        labels=None,
        scores=scores,
        boxes=None,
    )
    return instances


def crop_and_predict(
    predictor: SAM2ImagePredictor,
    image: Image.Image,
    crop_box: Box2D,
    point_coords: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    multimask_output: bool = False,
    device: str = "cuda",
) -> list[Instance2D] | Instance2D:
    """
    Crop the input image using the provided bounding box and then predict the segmentation mask for the cropped image using the provided model.

    Args:
        predictor: SAM2ImagePredictor instance used for making predictions.
        image: The input image for which to predict the segmentation mask.
        crop_box: The bounding box for cropping the image before prediction. The coordinates should be **pixel values** in the xyxy format.
        point_coords: The coordinates of the points for interactive segmentation.
        point_labels: The labels of the points for interactive segmentation.
        device: The device to run the model on. Default is "cuda". Can be set to "cpu" for CPU inference.

    Returns:
        A list of Instance2D objects representing the predicted instances.
    """
    # Crop the image using the provided bounding box
    x0, y0, x1, y1 = map(int, crop_box.xyxy)
    cropped_image = image.crop((x0, y0, x1, y1))

    # Convert point coordinates to the cropped image's coordinate system
    if point_coords is not None:
        point_coords_local = point_coords - np.array([[x0, y0]], dtype=np.float32)
        # Filter out points that are outside the cropped image
        valid_mask = (
            (point_coords_local[:, 0] >= 0)
            & (point_coords_local[:, 0] < cropped_image.width)
            & (point_coords_local[:, 1] >= 0)
            & (point_coords_local[:, 1] < cropped_image.height)
        )
        point_coords_local = point_coords_local[valid_mask]
        if point_labels is not None:
            point_labels = point_labels[valid_mask]

    # Inference and convert masks to Instance2D objects
    instances = predict(
        predictor=predictor,
        image=cropped_image,
        point_coords=point_coords_local if point_coords is not None else None,
        point_labels=point_labels if point_labels is not None else None,
        box=np.array([0, 0, cropped_image.width-1, cropped_image.height-1], dtype=np.int64),
        multimask_output=multimask_output,
        device=device,
    )
    # Convert the instances back to the original image's coordinate system
    instances = [instance.convert_to_original_coordinates(
            original_width=image.width,
            original_height=image.height,
            crop_xyxy=(x0, y0, x1, y1),
            input_width=cropped_image.width,
            input_height=cropped_image.height,
            normalized=False,
        ) for instance in instances]

    # If multimask_output is False, return the instances as a Instance2D object instead of a list
    if not multimask_output:
        return instances[0], cropped_image
    else:
        return instances, cropped_image
