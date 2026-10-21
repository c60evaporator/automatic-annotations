import numpy as np

from ..common.schemas import Box2D, Instance2D
from ..common.eval.detection import calc_box_ious
from ..common.eval.segmentation import calc_mask_ious

def retrieve_prompts_from_inference_state(
    inference_state: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Retrieves the prompt points, labels, and boxes from inference_state in the original image coordinate system.

    Args:
        inference_state: The current state of the video inference.

    Returns:
        A tuple containing:
            - A list of box prompts, where each box prompt is a dictionary containing:
                - "obj_id": The tracking ID of the object.
                - "frame_idx": The index of the frame.
                - "xyxy": The bounding box coordinates in xyxy format.
            - A list of point prompts, where each point prompt is a dictionary containing:
                - "obj_id": The tracking ID of the object.
                - "frame_idx": The index of the frame.
                - "points": The coordinates of the points in pixel values.
                - "labels": The labels of the points. 1 for positive points, 0 for negative points.
    """
    point_inputs_per_obj = inference_state["point_inputs_per_obj"]
    obj_idx_to_id = inference_state["obj_idx_to_id"]
    proc_height = inference_state["images"].shape[-2]
    proc_width = inference_state["images"].shape[-1]
    original_height = inference_state["video_height"]
    original_width = inference_state["video_width"]

    box_prompts = []
    point_prompts = []

    for obj_idx, obj_point_inputs in point_inputs_per_obj.items():
        obj_id = obj_idx_to_id[obj_idx]
        for frame_idx, frame_point_inputs in obj_point_inputs.items():
            # Retrieve point prompts
            point_coords = frame_point_inputs["point_coords"].cpu().numpy()[0]
            point_labels = frame_point_inputs["point_labels"].cpu().numpy()[0]
            # Convert point coordinates from processed image coordinate system to original image coordinate system
            point_coords[:, 0] = point_coords[:, 0] * (original_width / proc_width)
            point_coords[:, 1] = point_coords[:, 1] * (original_height / proc_height)

            box_topleft = None
            box_bottomright = None
            points = []
            labels = []
            for point_coord, point_label in zip(point_coords, point_labels):
                # Negative point
                if point_label == 0:
                    points.append(point_coord)
                    labels.append(0)
                # Positive point
                elif point_label == 1:
                    points.append(point_coord)
                    labels.append(1)
                # Box topleft
                elif point_label == 2:
                    if box_topleft is not None:
                        raise ValueError(f"Multiple box topleft points found for obj_id {obj_id} in frame {frame_idx}.")
                    box_topleft = point_coord
                # Box bottomright
                elif point_label == 3:
                    if box_bottomright is not None:
                        raise ValueError(f"Multiple box bottomright points found for obj_id {obj_id} in frame {frame_idx}.")
                    box_bottomright = point_coord
                else:
                    raise ValueError(f"Invalid point label {point_label} for obj_id {obj_id} in frame {frame_idx}. Must be 0, 1, 2, or 3.")
            # Build point prompt
            if len(points) > 0:
                point_prompts.append({
                    "obj_id": obj_id,
                    "frame_idx": frame_idx,
                    "points": np.stack(points, axis=0).astype(np.float32),
                    "labels": np.array(labels, dtype=np.int32),
                })

            # Build box prompt
            if (box_topleft is None) ^ (box_bottomright is None):
                raise ValueError(f"Only one of box_topleft or box_bottomright is set for obj_id {obj_id} in frame {frame_idx}. Both must be set or both must be None.")
            elif box_topleft is not None and box_bottomright is not None:
                box_prompts.append({
                    "obj_id": obj_id,
                    "frame_idx": frame_idx,
                    "xyxy": np.concatenate([box_topleft, box_bottomright], axis=0),
                })

    return box_prompts, point_prompts


def assign_continuous_tracking_ids(
    current_predicted_instances: list[Instance2D],
    prev_propagated_instances: list[Instance2D],
    max_track_id: int,
    iou_threshold: float,
    iou_method: str = "box",
    match_label: bool = True,
) -> int:
    """
    Assigns continuous tracking IDs to the current detected instances based on the IoUs with previous propagated instances.

    Args:
        current_predicted_instances: A list of Instance2D objects detected in the current frame.
        prev_propagated_instances: A list of Instance2D objects propagated from the previous frame.
        max_track_id: The maximum tracking ID assigned so far. New IDs will be assigned starting from max_track_id + 1.
        iou_threshold: The IoU threshold for matching instances. If the IoU between a current instance and a previous instance is above this threshold, they are considered the same object.
        iou_method: The method to compute IoU. Can be "box" or "mask". If "box", IoU is computed based on bounding boxes. If "mask", IoU is computed based on mask.
        match_label: If True, IoU matching will only be performed between instances with the same label. If False, IoU matching will be performed regardless of labels.

    Returns:
        A tuple containing:
            - A dictionary mapping the index of each current predicted instance to its assigned tracking ID.
            - The updated maximum tracking ID after assigning new IDs to unmatched instances.
    """
    if iou_method == "box":
        # instance.box is a prompt box. We want to use the box of the instance itself, not the prompt box. Therefore, we use instance.mask_region instead of instance.box.xyxy to compute IoU.
        prev_propagated_boxes = [Box2D(xyxy=np.array(instance.mask_region), label=instance.box.label, track_id=instance.box.track_id) 
                                 for instance in prev_propagated_instances]
        current_predicted_boxes = [Box2D(xyxy=np.array(instance.mask_region), label=instance.box.label, track_id=instance.box.track_id)
                                   for instance in current_predicted_instances]
        iou_matrix = calc_box_ious(current_predicted_boxes, prev_propagated_boxes, match_label=match_label)
    elif iou_method == "mask":
        iou_matrix = calc_mask_ious(current_predicted_instances, prev_propagated_instances, match_label=match_label)

    num_predicted_instances = len(current_predicted_instances)
    num_propagated_instances = len(prev_propagated_instances)

    # 全組み合わせをIoUの降順に並べる
    candidates: list[tuple[float, int, int]] = []

    for predicted_inst_idx in range(num_predicted_instances):
        for propagated_inst_idx in range(num_propagated_instances):
            iou = iou_matrix[predicted_inst_idx, propagated_inst_idx]

            if iou >= iou_threshold:
                candidates.append(
                    (float(iou), predicted_inst_idx, propagated_inst_idx)
                )

    candidates.sort(reverse=True)

    matched_frame_indices: set[int] = set()
    matched_prev_indices: set[int] = set()

    # IoUが高いペアから1対1で割り当てる
    idx_to_track_id: dict[int, int] = {}
    for iou, predicted_inst_idx, propagated_inst_idx in candidates:
        if predicted_inst_idx in matched_frame_indices:
            continue
        if propagated_inst_idx in matched_prev_indices:
            continue

        prev_track_id = prev_propagated_instances[propagated_inst_idx].box.track_id
        if prev_track_id is None:
            continue

        idx_to_track_id[predicted_inst_idx] = prev_track_id
        matched_frame_indices.add(predicted_inst_idx)
        matched_prev_indices.add(propagated_inst_idx)

    # マッチしなかった現在フレームのBoxには新規IDを発行
    for predicted_inst_idx in range(len(current_predicted_instances)):
        if predicted_inst_idx in matched_frame_indices:
            continue

        max_track_id += 1
        idx_to_track_id[predicted_inst_idx] = max_track_id
    
    return idx_to_track_id, max_track_id
