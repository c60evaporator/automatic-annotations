from typing import Optional
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import (
    Sam2Model,
    Sam2Processor,
    Sam2VideoModel,
    Sam2VideoProcessor,
    Sam2VideoInferenceSession,
)

from ..common.schemas import Box2D, Instance2D
from ..common.geometry.crop_resize import resize_mask_nearest

DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-large"


@dataclass
class Sam2ImagePredictor:
    """Set of the Transformers SAM2 model and processor for single image inference"""
    model: Sam2Model
    processor: Sam2Processor
    device: str


@dataclass
class Sam2VideoPredictor:
    """Set of the Transformers SAM2 model and processor for video inference"""
    model: Sam2VideoModel
    processor: Sam2VideoProcessor
    device: str
    dtype: torch.dtype


def load_image_predictor(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
) -> Sam2ImagePredictor:
    """
    Loads the SAM2 model and processor for single image inference from Hugging Face Hub.

    The model is loaded in float32 and run with bfloat16 autocast on CUDA (same as the official implementation).
    """
    model = Sam2Model.from_pretrained(model_id).to(device).eval()
    processor = Sam2Processor.from_pretrained(model_id)
    return Sam2ImagePredictor(model=model, processor=processor, device=device)


def load_video_predictor(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> Sam2VideoPredictor:
    """
    Loads the SAM2 model and processor for video inference from Hugging Face Hub.

    Args:
        model_id: Model ID on Hugging Face Hub.
        device: The device to run the model on.
        dtype: The dtype of the model and the inference state. Default is bfloat16 on CUDA and float32 on CPU.
    """
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Sam2VideoModel.from_pretrained(model_id).to(device, dtype=dtype).eval()
    processor = Sam2VideoProcessor.from_pretrained(model_id)
    return Sam2VideoPredictor(model=model, processor=processor, device=device, dtype=dtype)


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
    predictor: Sam2ImagePredictor,
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
        predictor: Sam2ImagePredictor created by load_image_predictor().
        image: The input image for which to predict the segmentation mask.
        point_coords: The coordinates of the points for interactive segmentation. Shape (N, 2) in pixel values.
        point_labels: The labels of the points for interactive segmentation. Shape (N,). 1 for positive points, 0 for negative points.
        box_coords: The bounding box for the object to segment. The coordinates should be **pixel values** in the xyxy format.
        multimask_output: Whether to output multiple masks per instance. Default is False.
        device: The device to run the model on. Default is "cuda". Can be set to "cpu" for CPU inference.

    Returns:
        A list of Instance2D objects representing the predicted instances.
    """
    image = image.convert("RGB")
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    # Convert the prompts into the nested list format of the processor: [image, object, point, xy]
    inputs = predictor.processor(
        images=image,
        input_points=None if point_coords is None else [[np.asarray(point_coords, dtype=np.float32).tolist()]],
        input_labels=None if point_labels is None else [[np.asarray(point_labels, dtype=np.int64).tolist()]],
        input_boxes=None if box_coords is None else [[np.asarray(box_coords, dtype=np.float32).tolist()]],
        return_tensors="pt",
    ).to(device)
    # Inference
    with torch.inference_mode(), autocast_context:
        outputs = predictor.model(**inputs, multimask_output=multimask_output)

    # Upscale the masks to the original image size. masks: (num_masks, H, W), scores: (num_masks,)
    masks = predictor.processor.post_process_masks(
        outputs.pred_masks.float().cpu(), inputs["original_sizes"].cpu()
    )[0][0].numpy().astype(np.uint8)
    scores = outputs.iou_scores[0, 0].float().cpu().numpy()

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
    predictor: Sam2ImagePredictor,
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
        predictor: Sam2ImagePredictor created by load_image_predictor().
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
def init_frame_state(
    predictor: Sam2VideoPredictor,
    images: list[Image.Image],
) -> Sam2VideoInferenceSession:
    """
    Initializes the frame state for video inference.

    Args:
        predictor: Sam2VideoPredictor created by load_video_predictor().
        images: The input images as a list of PIL.Image.Image.

    Returns:
        The inference state (Sam2VideoInferenceSession of Transformers) holding the preprocessed frames, prompts and tracking results.
    """
    inference_state = predictor.processor.init_video_session(
        video=[image.convert("RGB") for image in images],
        inference_device=predictor.device,
        dtype=predictor.dtype,
    )
    return inference_state


def reset_frame_state(
    predictor: Sam2VideoPredictor,
    inference_state: Sam2VideoInferenceSession,
) -> None:
    """
    Removes all the prompts and tracking results from the inference state, while keeping the video frames.

    Args:
        predictor: Sam2VideoPredictor created by load_video_predictor(). Not used, but kept for the same interface as the official implementation.
        inference_state: The inference state created by init_frame_state().
    """
    inference_state.reset_tracking_data()


def _get_video_size(inference_state: Sam2VideoInferenceSession) -> tuple[int, int]:
    return int(inference_state.video_height), int(inference_state.video_width)


def _postprocess_mask_logits(
    predictor: Sam2VideoPredictor,
    inference_state: Sam2VideoInferenceSession,
    pred_masks: torch.Tensor,
) -> np.ndarray:
    """Upscales the low-resolution mask logits to the original video size and binarizes them. Returns (N, H, W) bool array."""
    out_mask_logits = predictor.processor.post_process_masks(
        [pred_masks],
        original_sizes=[list(_get_video_size(inference_state))],
        binarize=False,
    )[0]  # (N, 1, H, W)
    return (out_mask_logits[:, 0] > 0.0).cpu().numpy()


def _run_inference_on_prompted_frame(
    predictor: Sam2VideoPredictor,
    inference_state: Sam2VideoInferenceSession,
    frame_idx: int,
    obj_id: int,
):
    """
    Runs the inference on a frame where new prompts are added to an object.

    This follows ``SAM2VideoPredictor.add_new_points_or_box`` of the official implementation, which the forward
    method of Transformers does not fully reproduce:

    - If the object already has a predicted mask on the frame (e.g. correction clicks on a tracked frame),
      the previous mask logits are fed into the mask decoder along with the new prompts, so that the clicks refine the existing mask.
    - The result is stored as a conditioning frame, so that the correction is kept and used as memory when the video
      is propagated again (equivalent to ``add_all_frames_to_correct_as_cond=True`` of the official implementation).
      Otherwise, the correction is overwritten by re-propagation.

    Returns:
        The outputs of the frame containing the masks of all the objects.
    """
    model = predictor.model
    obj_idx = inference_state.obj_id_to_idx(obj_id)
    obj_output_dict = inference_state.output_dict_per_obj[obj_idx]
    tracked_info = inference_state.frames_tracked_per_obj[obj_idx].get(frame_idx)
    is_init_cond_frame = tracked_info is None
    reverse = False if is_init_cond_frame else tracked_info["reverse"]

    # Get the previously predicted mask logits of the object on this frame, if any
    prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
    if prev_out is None:
        prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
    prev_sam_mask_logits = None
    if prev_out is not None and prev_out.get("pred_masks") is not None:
        prev_sam_mask_logits = torch.clamp(
            prev_out["pred_masks"].to(inference_state.inference_device), -32.0, 32.0
        )

    with torch.inference_mode():
        current_out = model._run_single_frame_inference(
            inference_session=inference_state,
            frame_idx=frame_idx,
            obj_idx=obj_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=inference_state.point_inputs_per_obj[obj_idx].get(frame_idx),
            mask_inputs=inference_state.mask_inputs_per_obj[obj_idx].get(frame_idx),
            reverse=reverse,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Store the result as a conditioning frame and encode it into the memory
        obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        inference_state.store_output(obj_idx, frame_idx, output_value=current_out, is_conditioning_frame=True)
        model._batch_encode_memories(
            inference_session=inference_state,
            frame_idx=frame_idx,
            objects_needing_memory_encoding=[obj_idx],
            high_res_masks_for_memory=[current_out["high_res_masks"]],
            object_score_logits_for_memory=[current_out["object_score_logits"]],
            is_mask_from_pts_per_obj=[True],
        )
    if obj_id in inference_state.obj_with_new_inputs:
        inference_state.obj_with_new_inputs.remove(obj_id)

    # Collect the masks of all the objects on the frame (the prompted object reuses the stored conditioning output)
    return model(inference_session=inference_state, frame_idx=frame_idx, reverse=reverse)


def _select_instances_by_obj_ids(
    predictor: Sam2VideoPredictor,
    inference_state: Sam2VideoInferenceSession,
    outputs,
    obj_ids: list[int],
    labels: list[str | None] | None = None,
    boxes: list[np.ndarray] | None = None,
) -> list[Instance2D]:
    """Converts the outputs of the frame to Instance2D objects in the order of ``obj_ids``."""
    out_masks = _postprocess_mask_logits(predictor, inference_state, outputs.pred_masks)  # (N, H, W)
    out_obj_ids = list(outputs.object_ids)
    indices = [out_obj_ids.index(obj_id) for obj_id in obj_ids]
    video_height, video_width = _get_video_size(inference_state)
    return masks_to_instances(
        masks=out_masks[indices],
        image_height=video_height,
        image_width=video_width,
        labels=labels,
        track_ids=obj_ids,
        scores=None,
        boxes=boxes,
    )


def add_box_prompts(
    predictor: Sam2VideoPredictor,
    inference_state: Sam2VideoInferenceSession,
    frame_idx: int,
    box_prompts: list[Box2D],
    point_prompts: Optional[list[dict]] = None,
) -> list[Instance2D]:
    """
    Adds bounding box prompts for a specific frame in the video.

    Args:
        predictor: Sam2VideoPredictor created by load_video_predictor().
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
        A list of Instance2D objects representing the predicted instances for the added box prompts, in the same order as ``box_prompts``.
    """
    if point_prompts is None:
        point_prompts = []

    outputs = None
    for box in box_prompts:
        obj_id = box.track_id
        obj_point_prompts = [point_prompt for point_prompt in point_prompts if point_prompt.get('obj_id') == obj_id]
        if len(obj_point_prompts) > 0:
            # Nested list format of the processor: [image, object, point, xy]
            points = [[[[float(v) for v in point_prompt['point']] for point_prompt in obj_point_prompts]]]
            labels = [[[int(point_prompt['label']) for point_prompt in obj_point_prompts]]]
        else:
            points = None
            labels = None
        # Add the prompts to the inference state and run the inference on the frame.
        # The inference is run for each object, because the processor only keeps the last added objects as new inputs.
        predictor.processor.add_inputs_to_inference_session(
            inference_session=inference_state,
            frame_idx=frame_idx,
            obj_ids=obj_id,
            input_points=points,
            input_labels=labels,
            input_boxes=[[np.asarray(box.xyxy, dtype=np.float32).tolist()]],
        )
        outputs = _run_inference_on_prompted_frame(predictor, inference_state, frame_idx, obj_id)

    if outputs is None:
        return []

    # Only the last outputs (containing the masks of all the objects in the frame) are used
    return _select_instances_by_obj_ids(
        predictor,
        inference_state,
        outputs,
        obj_ids=[box.track_id for box in box_prompts],
        labels=[box.label for box in box_prompts], # Use the labels from the box prompts if available
        boxes=[np.array(box.xyxy, dtype=np.float32) for box in box_prompts],
    )


def add_point_prompts(
    predictor: Sam2VideoPredictor,
    inference_state: Sam2VideoInferenceSession,
    frame_idx: int,
    point_prompts: list[dict],
) -> list[Instance2D]:
    """
    Adds point prompts for a specific frame in the video, e.g. correction clicks on a frame that has already been tracked.

    The corrected frame is kept as a conditioning frame, so the correction is not overwritten when propagate_inference() is called again.
    Note that the existing prompts of the object on the frame (including a box prompt) are replaced with the new point prompts.

    Args:
        predictor: Sam2VideoPredictor created by load_video_predictor().
        inference_state: The current state of the video inference.
        frame_idx: The index of the frame to which the point prompts should be added.
        point_prompts: The list of point prompts. The format of each point prompt dictionary should be:
        {
            'point': [x, y],  # The coordinates of the point in pixel values.
            'label': int,  # The label of the point. 1 for positive points, 0 for negative points.
            'obj_id': int  # representing the object ID associated with these points.
        }

    Returns:
        A list of Instance2D objects representing the predicted instances for the objects in ``point_prompts``.
    """
    obj_ids = list(dict.fromkeys(point_prompt['obj_id'] for point_prompt in point_prompts))
    outputs = None
    for obj_id in obj_ids:
        obj_point_prompts = [point_prompt for point_prompt in point_prompts if point_prompt['obj_id'] == obj_id]
        predictor.processor.add_inputs_to_inference_session(
            inference_session=inference_state,
            frame_idx=frame_idx,
            obj_ids=obj_id,
            input_points=[[[[float(v) for v in point_prompt['point']] for point_prompt in obj_point_prompts]]],
            input_labels=[[[int(point_prompt['label']) for point_prompt in obj_point_prompts]]],
        )
        outputs = _run_inference_on_prompted_frame(predictor, inference_state, frame_idx, obj_id)

    if outputs is None:
        return []

    return _select_instances_by_obj_ids(predictor, inference_state, outputs, obj_ids=obj_ids)


def propagate_inference(
    predictor: Sam2VideoPredictor,
    inference_state: Sam2VideoInferenceSession,
    start_frame_idx: int | None = None,
    max_frame_num_to_track=None,
    reverse=False,
) -> dict[int, dict[int, Instance2D]]:
    """
    Inference propagation for all objects across all frames in the video.

    Args:
        predictor: Sam2VideoPredictor created by load_video_predictor().
        inference_state: The inference state created by init_frame_state() and updated by add_box_prompts() to add prompts.
        start_frame_idx: The frame index to start the propagation. If None, the earliest frame with prompts is used.
        max_frame_num_to_track: The maximum number of frames to track. If None, all the frames are tracked.
        reverse: Whether to propagate in reverse order.

    Returns:
        A dictionary mapping the frame index to a dictionary of {obj_id: Instance2D}.
    """
    result_instances = {}
    video_height, video_width = _get_video_size(inference_state)
    # run propagation throughout the video
    for outputs in predictor.model.propagate_in_video_iterator(
        inference_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=reverse,
    ):
        out_frame_idx = outputs.frame_idx
        result_instances[out_frame_idx] = {}
        # Convert masks to Instance2D objects
        out_masks = _postprocess_mask_logits(predictor, inference_state, outputs.pred_masks)  # (N, H, W)
        predicted_instances = masks_to_instances(
            masks=out_masks,
            image_height=video_height,
            image_width=video_width,
            labels=None,
            track_ids=outputs.object_ids,
            scores=None,
            boxes=None,  # Box prompts are not associated with each frame, so we don't have box prompts for each frame. Therefore, we set boxes to None.
        )
        for predicted_instance in predicted_instances:
            out_obj_id = predicted_instance.box.track_id
            result_instances[out_frame_idx][out_obj_id] = predicted_instance

    return result_instances
