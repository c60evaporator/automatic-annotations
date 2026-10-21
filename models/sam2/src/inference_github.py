from typing import Optional
from contextlib import nullcontext
from collections import OrderedDict

import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from ..common.schemas import Box2D, Instance2D
from ..common.geometry.crop_resize import resize_mask_nearest

def masks_to_instances(
    masks: np.ndarray,
    image_height: int,
    image_width: int,
    labels: list[str] | None = None,
    track_ids: list[int] | None = None,
    scores: np.ndarray | None = None,
    boxes: np.ndarray | None = None,
    threshold: float = 0.5,
) -> list[Instance2D]:
    """
    Converts the predicted masks to a list of Instance2D objects.

    Args:
        masks: The predicted masks as a NumPy array of shape (C, H, W). The values should be 0 or 1 of uint8 or float32 type.
        image_height: The height of the image before resizing.
        image_width: The width of the image before resizing.
        labels: The labels for each instance.
        track_ids: The tracking IDs for each instance.
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
                    track_id=None if track_ids is None else int(track_ids[index]),
                    score=None if scores is None else float(scores[index]),
                ),
                mask=mask,
                mask_region=(rx0, ry0, rx1, ry1),
            )
        )
    return instances


####### Single Image Inference #######
def predict(
    predictor: SAM2ImagePredictor,
    image: Image.Image,
    point_coords: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    box_coords: np.ndarray | None = None,
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
        box_coords: The bounding box for the object to segment. The coordinates should be **pixel values** in the xyxy format.
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
            box=box_coords,
            multimask_output=multimask_output,
        )

    # Sort masks by scores in descending order
    if multimask_output:
        sorted_indices = np.argsort(scores)[::-1]
        masks = masks[sorted_indices]
        scores = scores[sorted_indices]

    # Convert masks to Instance2D objects
    boxes = np.stack([box_coords] * len(masks)) if box_coords is not None else None
    instances = masks_to_instances(
        masks=masks,
        image_height=image.height,
        image_width=image.width,
        labels=None,
        track_ids=None,
        scores=scores,
        boxes=boxes, # Save box prompt as the ``box`` attribute of the Instance2D object.
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
        box_coords=np.array([0, 0, cropped_image.width-1, cropped_image.height-1], dtype=np.int64),
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

    # If no instances were detected, return None
    if not instances:
        return None, cropped_image

    # If multimask_output is False, return the instances as a Instance2D object instead of a list
    if not multimask_output:
        return instances[0], cropped_image
    else:
        return instances, cropped_image


###### Video Inference ######
def _load_video_frames(
    images: list[Image.Image],
    image_size,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
) -> tuple[torch.Tensor, int, int]:
    """
    Loads and preprocesses image frames for video inference.

    Args:
        images: A list of PIL.Image.Image objects representing the video frames.
        image_size: The size to which the images should be resized.
        img_mean: The mean for normalization.
        img_std: The standard deviation for normalization.
        compute_device: The device to which the images should be transferred.
    Returns:
        A tuple containing:
            - A tensor of shape (N, C, H, W) containing the preprocessed images.
            - The original height of the images.
            - The original width of the images.
    """
    # PIL -> Tensor
    cpu_image_transform = v2.Compose([
        v2.ToImage(),
    ])
    images = [cpu_image_transform(img) for img in images]
    images = torch.stack(images)  # (N, C, H, W)
    original_height, original_width = images.shape[-2], images.shape[-1]
    # resize and normalize
    cpu_batch_transform = v2.Compose([
        v2.Resize(size=(image_size, image_size), antialias=True),
    ])
    device_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=img_mean, std=img_std),
    ])
    images = cpu_batch_transform(images)
    images = images.pin_memory()  # Pin memory for faster transfer to GPU
    images = images.to(compute_device, non_blocking=True)  # Transfer to GPU. `non_blocking=True`` allows for asynchronous transfer, which can improve performance when using pinned memory.
    images = device_transform(images)

    return images, original_height, original_width


def init_frame_state(
    predictor: SAM2VideoPredictor,
    images: list[Image.Image],
) -> dict:
    """
    Initializes the frame state for video inference.

    Args:
        predictor: An instance of SAM2VideoPredictor.
        images: The input images as a list of PIL.Image.Image.
        device: The device to run the model on. If None, uses the predictor's device.
    """
    offload_video_to_cpu=False
    offload_state_to_cpu=False
    # Load and preprocess the video frames
    image_tensors, video_height, video_width = _load_video_frames(
        images=images,
        image_size=predictor.image_size,
        compute_device=predictor.device,
    )
    # Store the inference state
    inference_state = {}
    inference_state["images"] = image_tensors
    inference_state["num_frames"] = len(images)
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu # whether to offload the video frames to CPU memory
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu # whether to offload the inference state to CPU memory
    inference_state["video_height"] = video_height # Original height of the video frames
    inference_state["video_width"] = video_width # Original width of the video frames
    inference_state["device"] = predictor.device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = predictor.device  # `offload_state_to_cpu` is always False, so the storage device is always the predictor's device
    inference_state["storage_device"] = predictor.device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    inference_state["frames_tracked_per_obj"] = {}
    # Warm up the visual backbone and cache the image feature on frame 0
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    
    return inference_state

def add_box_prompts(
    predictor: SAM2VideoPredictor,
    inference_state: dict,
    frame_idx: int,
    box_prompts: list[Box2D],
    point_prompts: Optional[list[dict]] = None,
) -> list[Instance2D]:
    """
    Adds bounding box prompts for a specific frame in the video.

    Args:
        predictor: An instance of SAM2VideoPredictor.
        inference_state: The current state of the video inference.
        frame_idx: The index of the frame to which the box prompt should be added.
        box_prompts: The list of bounding boxes to add as prompts, in xyxy format. **``track_id`` attribute is used as the object ID for tracking.**
        point_prompts: An optional list of point prompts to add alongside the box prompts. Each element should be a dictionary containing 'points' and 'labels'.

        The format of each point prompt dictionary should be:
        {
            'point': [x, y],  # The coordinates of the point in pixel values.
            'label': int,  # The label of the point. 1 for positive points, 0 for negative points.
            'obj_id': int  # representing the object ID associated with these points. This is useful for tracking objects across frames.
        }

    Returns:
        A list of Instance2D objects representing the predicted instances for the added box prompts.
    """
    if point_prompts is None:
        point_prompts = []

    for box in box_prompts:
        obj_id = box.track_id
        obj_point_prompts = [point_prompt for point_prompt in point_prompts if point_prompt.get('obj_id') == obj_id]
        if len(obj_point_prompts) > 0:
            points = [point_prompt['point'] for point_prompt in obj_point_prompts]
            labels = [point_prompt['label'] for point_prompt in obj_point_prompts]
        else:
            points = None
            labels = None
        # Convert box to numpy array
        box_array = np.array(box.xyxy, dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box_array,
        )

    # Only the last out_obj_ids and out_mask_logits are used
    out_masks = (np.squeeze(out_mask_logits, axis=1) > 0.0).cpu().numpy()  # (N, H, W)

    # Convert masks to Instance2D objects
    predicted_instances = masks_to_instances(
        masks=out_masks,
        image_height=inference_state["video_height"],
        image_width=inference_state["video_width"],
        labels=[box.label for box in box_prompts] if box_prompts is not None else None, # Use the labels from the box prompts if available
        track_ids=out_obj_ids,
        scores=None,
        boxes=[np.array(box.xyxy, dtype=np.float32) for box in box_prompts] if box_prompts is not None else None,
    )

    return predicted_instances


def propagate_inference(
    predictor: SAM2VideoPredictor,
    inference_state: dict,
    start_frame_idx: int = 0,
    max_frame_num_to_track=None,
    reverse=False,
) -> dict[int, list[dict]]:
    """
    Inference propagation for all objects across all frames in the video.

    Args:
        predictor: An instance of SAM2VideoPredictor.
        inference_state: The inference state created by init_frame_state() and updated by add_box_prompts() to add prompts.

    Returns:
        A list of dictionaries, each containing the predicted instances for a specific frame in the video.
    
        Each dictionary in the list contains:
            - "obj_id": The tracking ID of the object.
            - "frame_idx": The index of the frame.
            - "instance": The Instance2D object representing the predicted instance for that frame.
    """
    result_instances = {}
    # run propagation throughout the video
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=reverse,
    ):
        result_instances[out_frame_idx] = {}
        # Convert masks to Instance2D objects
        out_masks = (np.squeeze(out_mask_logits, axis=1) > 0.0).cpu().numpy()  # (N, H, W)
        predicted_instances = masks_to_instances(
            masks=out_masks,
            image_height=inference_state["video_height"],
            image_width=inference_state["video_width"],
            labels=None,
            track_ids=out_obj_ids,
            scores=None,
            boxes=None,  # Box prompts are note associated with each frame, so we don't have box prompts for each frame. Therefore, we set boxes to None.
        )
        for predicted_instance in predicted_instances:
            out_obj_id = predicted_instance.box.track_id
            result_instances[out_frame_idx][out_obj_id] = predicted_instance

    return result_instances
