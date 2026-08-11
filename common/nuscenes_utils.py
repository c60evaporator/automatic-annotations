import numpy as np

from .schemas import Box3D


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


###### 3D bounding box transformation functions ######
def get_sample_data_bboxes(sample_data: dict[str, dict],
                           sample_annotations: dict[str, dict],
                           instances: dict[str, dict],
                           categories: dict[str, dict],
                           category_conversion: dict[str, str] = None,
                           delete_unspecified_categories: bool = True,
                           tracking_ids: dict[str, int] = None):
    """
    Get bounding boxes in the camera's field of view for each sample data entry.

    Args:
        sample_data (dict): Dictionary of sample data entries. The keys are sample data tokens and the values are dict from sample_data.json.
        sample_annotations (dict): Dictionary of sample annotations. The keys are sample annotation tokens and the values are dict from sample_annotation.json.
        instances (dict): Dictionary of instances. The keys are instance tokens and the values are dict from instance.json.
        categories (dict): Dictionary of categories. The keys are category tokens and the values are dict from category.json.
        category_conversion (dict): Dictionary for converting category names to desired labels. If None, the original category names will be used.
        delete_unspecified_categories (bool): If True, boxes with categories not in category_conversion will be deleted. If False, they will be kept with their original category names.
        tracking_ids (dict): Dictionary for converting instance tokens to tracking IDs. If None, tracking IDs will be set to None.
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
        track_id=tracking_ids[sa["instance_token"]] if tracking_ids is not None else None
    ) for sa in annotations_in_sample]
    print(f"Number of boxes in sample {sample_data['sample_token']}: {len(boxes_3d)}")
    # Delete boxes with unspecified categories if required
    if delete_unspecified_categories:
        boxes_3d = [box for box in boxes_3d if box.label is not None]
        print(f"Number of boxes after removing unspecified categories: {len(boxes_3d)}")

    return boxes_3d
    
