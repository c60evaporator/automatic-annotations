import numpy as np

from .schemas import Box3D, Box2D
from .geometry.detection import (
    convert_global_bbox_to_ego,
    filter_boxes_in_camera_fov,
)
from .image_processing.detection import convert_3d_box_to_2d_box

# Category mapping from Nuscenes to UniAD (Based on MMdet3D v1.0.0rc6. Referenece https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc6/mmdet3d/datasets/nuscenes_dataset.py#L56)
CATEGORY_MAPPING_TO_UNIAD = {
    'vehicle.car': {'category_name': 'car', 'id': 0, 'category_group': 'vehicle'},
    'vehicle.truck': {'category_name': 'truck', 'id': 1, 'category_group': 'vehicle'},
    'vehicle.construction': {'category_name': 'construction_vehicle', 'id': 2, 'category_group': 'vehicle'},
    'vehicle.bus.bendy': {'category_name': 'bus', 'id': 3, 'category_group': 'vehicle'},
    'vehicle.bus.rigid': {'category_name': 'bus', 'id': 3, 'category_group': 'vehicle'},
    'vehicle.trailer': {'category_name': 'trailer', 'id': 4, 'category_group': 'vehicle'},
    'movable_object.barrier': {'category_name': 'barrier', 'id': 5, 'category_group': 'road_object'},
    'vehicle.motorcycle': {'category_name': 'motorcycle', 'id': 6, 'category_group': 'two_wheeler'},
    'vehicle.bicycle': {'category_name': 'bicycle', 'id': 7, 'category_group': 'two_wheeler'},
    'human.pedestrian.adult': {'category_name': 'pedestrian', 'id': 8, 'category_group': 'pedestrian'},
    'human.pedestrian.child': {'category_name': 'pedestrian', 'id': 8, 'category_group': 'pedestrian'},
    'human.pedestrian.construction_worker': {'category_name': 'pedestrian', 'id': 8, 'category_group': 'pedestrian'},
    'human.pedestrian.police_officer': {'category_name': 'pedestrian', 'id': 8, 'category_group': 'pedestrian'},
    'movable_object.trafficcone': {'category_name': 'traffic_cone', 'id': 9, 'category_group': 'road_object'},
}


def get_scene_contents(scene_token: str,
                       samples_all: list[dict],
                       sample_data_all: list[dict],
                       ego_poses_all: list[dict],
                       calibrated_sensors_all: list[dict],
                       sample_annotations_all: list[dict] | None = None,
                       instances_all: list[dict] | None = None) -> dict[str, dict]:
    """
    Get all the contents of a scene from the Nuscenes dataset as a dictionary whose keys are their tokens.

    Args:
        scene_token (str): The token of the scene to retrieve.
        samples_all (list[dict]): List of all samples in the dataset.
        sample_data_all (list[dict]): List of all sample data entries in the dataset.
        ego_poses_all (list[dict]): List of all ego poses in the dataset.
        calibrated_sensors_all (list[dict]): List of all calibrated sensors in the dataset.
        sample_annotations_all (list[dict]): List of all sample annotations in the dataset. If None, the sample_annotations key will not be included in the returned dictionary.
        instances_all (list[dict]): List of all instances in the dataset. If None, the instances key will not be included in the returned dictionary.

    Returns:
        dict: A dictionary containing all the contents of the scene.

    Note:
        The returned dictionary contains the following keys:
        - 'samples': A dictionary of sample dictionaries, keyed by sample token and sorted by timestamp.
        - 'sample_data': A dictionary of sample data dictionaries that is filtered by is_key_frame, keyed by sample data token and sorted by sample order.
        - 'ego_poses': A dictionary of ego pose dictionaries, keyed by ego pose token. NOT sorted.
        - 'calibrated_sensors': A dictionary of calibrated sensor dictionaries, keyed by calibrated sensor token. NOT sorted.
        - 'sample_annotations': A dictionary of sample annotation dictionaries, keyed by sample annotation token. NOT sorted.
        - 'instances': A dictionary of instance dictionaries, keyed by instance token. NOT sorted.
    """
    if (sample_annotations_all is None) ^ (instances_all is None):
        raise ValueError("Both sample_annotations_all and instances_all must be provided or both must be None.")

    # Select the samples and sort them by timestamp
    samples = {sample["token"]: sample for sample in samples_all 
               if sample["scene_token"] == scene_token}
    samples = dict(sorted(samples.items(), key=lambda item: item[1]["timestamp"]))
    sample_order = {sample_token: i for i, sample_token in enumerate(samples.keys())}
    # Select the sample_data in the scene and filter by is_key_frame, then sort them by sample order
    sample_data = {sd["token"]: sd for sd in sample_data_all if sd["sample_token"] in samples.keys()}
    sample_data = {sd_token: sd for sd_token, sd in sample_data.items() if sd["is_key_frame"]}  # Filter by is_key_frame
    sample_data = dict(sorted(sample_data.items(), key=lambda item: sample_order[item[1]["sample_token"]]))  # Sort by sample order
    # Select the ego_poses and calibrated_sensors in the scene and sort them by sample_data order
    ego_pose_tokens = set(dict.fromkeys(sd["ego_pose_token"] for sd in sample_data.values()))
    ego_poses_by_token = {ep["token"]: ep for ep in ego_poses_all if ep["token"] in ego_pose_tokens}
    ego_poses = {token: ego_poses_by_token[token] for token in ego_pose_tokens}
    calibrated_sensor_tokens = set(dict.fromkeys(sd["calibrated_sensor_token"] for sd in sample_data.values()))
    calibrated_sensors_by_token = {cs["token"]: cs for cs in calibrated_sensors_all
                                   if cs["token"] in calibrated_sensor_tokens}
    calibrated_sensors = {token: calibrated_sensors_by_token[token]
                          for token in calibrated_sensor_tokens}

    result = {
        "samples": samples,
        "sample_data": sample_data,
        "ego_poses": ego_poses,
        "calibrated_sensors": calibrated_sensors,
    }

    if sample_annotations_all is not None and instances_all is not None:
        # Select the sample_annotations in the scene and sort them by sample order
        sample_annotations = {sa["token"]: sa for sa in sample_annotations_all if sa["sample_token"] in samples.keys()}
        sample_annotations = dict(sorted(sample_annotations.items(), key=lambda item: sample_order[item[1]["sample_token"]]))
        result["sample_annotations"] = sample_annotations
        # Select the instances in the scene and sort them by sample_annotation order
        instance_tokens = list(dict.fromkeys(sa["instance_token"] for sa in sample_annotations.values()))
        instances_by_token = {inst["token"]: inst for inst in instances_all if inst["token"] in set(instance_tokens)}
        instances = {token: instances_by_token[token] for token in instance_tokens}
        result["instances"] = instances

    return result

def get_sample_contents(sample_index: int,
                        samples_in_scene: dict[str, dict],
                        sample_data_in_scene: dict[str, dict],
                        ego_poses_in_scene: dict[str, dict],
                        calibrated_sensors_in_scene: dict[str, dict],
                        sensor_token: str | None = None) -> dict[str, dict]:
    """
    Get the contents of a sample from the Nuscenes dataset as a dictionary whose keys are their tokens.

    Args:
        sample_index (int): The index of the sample in the scene.
        samples_in_scene (dict): Dictionary of samples in the scene, keyed by sample token.
        sample_data_in_scene (dict): Dictionary of sample data in the scene, keyed by sample data token.
        ego_poses_in_scene (dict): Dictionary of ego poses in the scene, keyed by ego pose token.
        calibrated_sensors_in_scene (dict): Dictionary of calibrated sensors in the scene, keyed by calibrated sensor token.
        sample_annotations_in_scene (dict): Dictionary of sample annotations in the scene, keyed by sample annotation token. If None, the sample_annotations key will not be included in the returned dictionary.
        instances_in_scene (dict): Dictionary of instances in the scene, keyed by instance token. If None, the instances key will not be included in the returned dictionary.
        sensor_token (str): The token of the sensor to filter the sample data. If None, all sample data will be included.

    Returns:
        dict: Dictionary containing the contents of the specified sample.
    """
    sample_token = list(samples_in_scene.keys())[sample_index]
    sample_data = {sd_token: sd for sd_token, sd in sample_data_in_scene.items() if sd["sample_token"] == sample_token}
    calibrated_sensor_tokens = list(dict.fromkeys(sd["calibrated_sensor_token"] for sd in sample_data.values()))
    calibrated_sensors = {token: calibrated_sensors_in_scene[token] for token in calibrated_sensor_tokens}
    ego_pose_tokens = set(sd["ego_pose_token"] for sd in sample_data.values())
    ego_poses = {ep_token: ep for ep_token, ep in ego_poses_in_scene.items() if ep_token in ego_pose_tokens}

    if sensor_token is None:
        result = {
            "sample": samples_in_scene[sample_token],
            "sample_data": sample_data,
            "ego_poses": ego_poses,
            "calibrated_sensors": calibrated_sensors,
        }

    # Filter sample_data and calibrated_sensors by sensor_token if provided
    if sensor_token is not None:
        calibrated_sensor = next(cs for cs in calibrated_sensors.values() if cs["sensor_token"] == sensor_token)
        sample_data = next(sd for sd in sample_data.values() if sd["calibrated_sensor_token"] == calibrated_sensor["token"])
        ego_pose = ego_poses[sample_data["ego_pose_token"]]
        result = {
            "sample": samples_in_scene[sample_token],
            "sample_data": sample_data,
            "ego_pose": ego_pose,
            "calibrated_sensor": calibrated_sensor,
        }

    return result


def get_sample_window_contents(window_start: int,
                               window_end: int,
                               samples_in_scene: dict[str, dict],
                               sample_data_in_scene: dict[str, dict],
                               ego_poses_in_scene: dict[str, dict],
                               calibrated_sensors_in_scene: dict[str, dict],
                               sensor_token: str | None = None) -> dict[str, dict]:
    """
    Get the contents of a sample window from the Nuscenes dataset as a dictionary whose keys are their tokens.

    Args:
        window_start (int): The start index of the sample window in the scene.
        window_end (int): The end index of the sample window in the scene.
        samples_in_scene (dict): Dictionary of samples in the scene, keyed by sample token.
        sample_data_in_scene (dict): Dictionary of sample data in the scene, keyed by sample data token.
        ego_poses_in_scene (dict): Dictionary of ego poses in the scene, keyed by ego pose token.
        calibrated_sensors_in_scene (dict): Dictionary of calibrated sensors in the scene, keyed by calibrated sensor token.
        sensor_token (str): The token of the sensor to filter the sample data. If None, all sample data will be included.

    Returns:
        dict: Dictionary containing the contents of the specified sample.
    """
    window_samples = dict(list(samples_in_scene.items())[window_start:window_end+1])
    window_sample_order = {sample_token: i for i, sample_token in enumerate(window_samples)}

    if sensor_token is not None:
        calibrated_sensors = {cs_token: cs for cs_token, cs in calibrated_sensors_in_scene.items() if cs["sensor_token"] == sensor_token}
        sample_data = {sd_token: sd for sd_token, sd in sample_data_in_scene.items() if sd["calibrated_sensor_token"] in calibrated_sensors}
        ego_poses = {ep_token: ep for ep_token, ep in ego_poses_in_scene.items() if ep_token in set(sd["ego_pose_token"] for sd in sample_data.values())}
    else:
        sample_data = sample_data_in_scene
        ego_poses = ego_poses_in_scene
        calibrated_sensors = calibrated_sensors_in_scene

    window_sample_data = {sd_token: sd for sd_token, sd in sample_data.items() if sd["sample_token"] in window_sample_order}
    window_sample_data = dict(sorted(window_sample_data.items(), key=lambda item: window_sample_order[item[1]["sample_token"]]))
    window_ego_poses_tokens = list(dict.fromkeys(sd["ego_pose_token"] for sd in window_sample_data.values()))
    window_ego_poses = {ep_token: ego_poses[ep_token] for ep_token in window_ego_poses_tokens}
    window_calibrated_sensor_tokens = list(dict.fromkeys(sd["calibrated_sensor_token"] for sd in window_sample_data.values()))
    window_calibrated_sensors = {cs_token: calibrated_sensors[cs_token] for cs_token in window_calibrated_sensor_tokens}

    return {
        "samples": window_samples,
        "sample_data": window_sample_data,
        "ego_poses": window_ego_poses,
        "calibrated_sensors": window_calibrated_sensors,
    }


def get_annotations_in_sample(sample_index: int,
                              samples_in_scene: dict[str, dict],
                              sample_annotations_in_scene: dict[str, dict] | None = None,
                              instances_in_scene: dict[str, dict] | None = None) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Get the sample_annotations and instances in a sample from the Nuscenes dataset.

    Args:
        sample_index (int): The index of the sample in the scene.
        samples_in_scene (dict): Dictionary of samples in the scene, keyed by sample token.
        sample_annotations_in_scene (dict): Dictionary of sample annotations in the scene, keyed by sample annotation token. If None, the sample_annotations key will not be included in the returned dictionary.
        instances_in_scene (dict): Dictionary of instances in the scene, keyed by instance token. If None, the instances key will not be included in the returned dictionary.

    Returns:
        tuple[dict[str, dict], dict[str, dict]]: Tuple containing the sample annotations and instances in the specified sample.
    """
    sample_token = list(samples_in_scene.keys())[sample_index]

    sample_annotations = {sa_token: sa for sa_token, sa in sample_annotations_in_scene.items() if sa["sample_token"] == sample_token}

    instance_tokens = set(ann["instance_token"] for ann in sample_annotations.values())
    instances = {token: inst for token, inst in instances_in_scene.items() if token in instance_tokens}        

    return sample_annotations, instances


###### 3D bounding box transformation functions ######
def get_sample_data_bboxes(sample_data: dict[str, dict],
                           sample_annotations: dict[str, dict],
                           instances: dict[str, dict],
                           categories: dict[str, dict],
                           category_conversion: dict[str, str] = None,
                           delete_unspecified_categories: bool = True,
                           track_ids: dict[str, int] = None):
    """
    Get bounding boxes in the camera's field of view for each sample data entry.

    Args:
        sample_data (dict): Dictionary of sample data entries. The keys are sample data tokens and the values are dict from sample_data.json.
        sample_annotations (dict): Dictionary of sample annotations. The keys are sample annotation tokens and the values are dict from sample_annotation.json.
        instances (dict): Dictionary of instances. The keys are instance tokens and the values are dict from instance.json.
        categories (dict): Dictionary of categories. The keys are category tokens and the values are dict from category.json.
        category_conversion (dict): Dictionary for converting category names to desired labels. If None, the original category names will be used.
        delete_unspecified_categories (bool): If True, boxes with categories not in category_conversion will be deleted. If False, they will be kept with their original category names.
        track_ids (dict): Dictionary for converting instance tokens to tracking IDs. If None, tracking IDs will be set to None.
    """
    if category_conversion is None:
        category_conversion = {cat["name"]: cat["name"] for cat in categories.values()}

    # Getting bounding boxes in the sample
    annotations_in_sample = [sa for sa in sample_annotations.values() if sa["sample_token"] == sample_data["sample_token"]]
    boxes_3d = [Box3D.from_dimensions(
        center=np.array(sa["translation"]),
        length=sa["size"][1],
        width=sa["size"][0],
        height=sa["size"][2],
        rotation=np.array(sa["rotation"]),
        label=category_conversion.get(categories[instances[sa["instance_token"]]["category_token"]]["name"], None),
        track_id=track_ids[sa["instance_token"]] if track_ids is not None else None
    ) for sa in annotations_in_sample]
    print(f"Number of boxes in sample {sample_data['sample_token']}: {len(boxes_3d)}")
    # Delete boxes with unspecified categories if required
    if delete_unspecified_categories:
        boxes_3d = [box for box in boxes_3d if box.label is not None]
        print(f"Number of boxes after removing unspecified categories: {len(boxes_3d)}")

    return boxes_3d
    
def get_sample_data_2d_bboxes(
    sample_data: dict[str, dict],
    sample_annotations: dict[str, dict],
    instances: dict[str, dict],
    categories: dict[str, dict],
    calibrated_sensors: dict[str, dict],
    ego_poses: dict[str, dict],
    category_conversion: dict[str, str] = None,
    delete_unspecified_categories: bool = True,
    track_ids: dict[str, int] = None
) -> tuple[list[Box3D], list[Box2D]]:
    """
    Get bounding boxes in the camera's field of view for each sample data entry and convert them into 2D bounding boxes

    Args:
        sample_data (dict): Dictionary of sample data entries. The keys are sample data tokens and the values are dict from sample_data.json.
        ego_poses (dict): Dictionary of ego poses. The keys are ego pose tokens and the values are dict from ego_pose.json.
        calibrated_sensors (dict): Dictionary of calibrated sensors. The keys are calibrated sensor tokens and the values are dict from calibrated_sensor.json.
        sample_annotations (dict): Dictionary of sample annotations. The keys are sample annotation tokens and the values are dict from sample_annotation.json.
        instances (dict): Dictionary of instances. The keys are instance tokens and the values are dict from instance.json.
        categories (dict): Dictionary of categories. The keys are category tokens and the values are dict from category.json.
        category_conversion (dict): Dictionary for converting category names to desired labels. If None, the original category names will be used.
        delete_unspecified_categories (bool): If True, boxes with categories not in category_conversion will be deleted. If False, they will be kept with their original category names.
        track_ids (dict): Dictionary for converting instance tokens to tracking IDs. If None, tracking IDs will be set to None.
    """
    # Get the ground truth bounding boxes
    ego_pose = ego_poses[sample_data["ego_pose_token"]]
    image_width = sample_data["width"]
    image_height = sample_data["height"]
    camera_translation = calibrated_sensors[sample_data["calibrated_sensor_token"]]["translation"]
    camera_rotation = calibrated_sensors[sample_data["calibrated_sensor_token"]]["rotation"]
    camera_intrinsic = calibrated_sensors[sample_data["calibrated_sensor_token"]]["camera_intrinsic"]
    boxes_3d = get_sample_data_bboxes(sample_data, sample_annotations, instances, categories,
                                      category_conversion=category_conversion,
                                      delete_unspecified_categories=delete_unspecified_categories,
                                      track_ids=track_ids)
    boxes_3d_ego = [convert_global_bbox_to_ego(box, ego_pose["translation"], ego_pose["rotation"]) for box in boxes_3d]
    valid_boxes_3d_ego = filter_boxes_in_camera_fov(boxes_3d_ego, camera_translation, camera_rotation,
                                                    camera_intrinsic, image_width, image_height)
    boxes_2d = [convert_3d_box_to_2d_box(box, camera_translation, camera_rotation, camera_intrinsic, image_width, image_height)
                for box in valid_boxes_3d_ego]
    boxes_2d_filtered = [box for box in boxes_2d if box is not None]
    
    return valid_boxes_3d_ego, boxes_2d_filtered

def filter_category_group_bboxes(
    boxes_3d: list[Box3D],
    boxes_2d: list[Box2D],
    category_group: str,
    category_mapping: dict[str, dict],
) -> tuple[set[str], list[Box3D], list[Box2D]]:
    """
    Filter 3D and 2D bounding boxes based on a specified category group.

    Args:
        boxes_3d (list[Box3D]): List of 3D bounding boxes.
        boxes_2d (list[Box2D]): List of 2D bounding boxes.
        category_group (str): The category group to filter by (e.g., 'vehicle', 'pedestrian').
        category_mapping (dict): A dictionary mapping category names to their corresponding group and other metadata.
    Returns:
        tuple[set[str], list[Box3D], list[Box2D]]: A tuple containing the set of category names in the group,
            the filtered list of 3D bounding boxes, and the filtered list of 2D bounding boxes.
    """
    category_names_in_group = set([v['category_name'] for v in category_mapping.values() 
                                   if v['category_group'] == category_group])
    filtered_boxes_3d = [box for box in boxes_3d if box.label in category_names_in_group]
    filtered_boxes_2d = [box for box in boxes_2d if box.label in category_names_in_group]
    return filtered_boxes_3d, filtered_boxes_2d
