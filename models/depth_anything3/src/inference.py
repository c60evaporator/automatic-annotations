import numpy as np

from ..common.geometry.crop_resize import scale_intrinsic
from ..common.geometry.depth import depth_map_to_point_cloud
from ..common.image_processing.segmentation import mask_morphology


def get_pseudo_lidar(depth_map: np.ndarray,
                     camera_intrinsics: np.ndarray,
                     original_image_width: int,
                     original_image_height: int,
                     model_name: str,
                     mask: np.ndarray | None = None,
                     morph_kernel_sizes: list[int] | None = None,
                     ratio_morphology: bool = False,
) -> np.ndarray:
    """
    Generate a pseudo-LiDAR point cloud from a depth map and camera intrinsics.

    Args:
        depth_map (np.ndarray): HxW array of depth values.
        camera_intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        original_image_width (int): Width of the original image before resizing.
        original_image_height (int): Height of the original image before resizing.
        model_name (str): Name of the depth estimation model used for identifying metric scale.
        mask (np.ndarray | None): HxW binary mask array. Only points where mask is True will be included. If None, all points are included.
        morph_kernel_sizes (list[int] | None): List of kernel sizes for morphological operations. Positive size represents dilation, whereas negative size represents erosion. If None, no morphological operations are applied.
        ratio_morphology (bool): Whether to apply ratio-based morphology. If True, the applied kernel sizes are computed by multiplying `morph_kernel_sizes` by the average of the mask bounding box height and width.

    Returns:
        np.ndarray: Nx3 array of 3D points in the camera coordinate system. Only points where mask is True are included.
    """
    # Scale the camera intrinsics to match the depth map size
    depth_map_height, depth_map_width = depth_map.shape
    scale_x = depth_map_width / original_image_width
    scale_y = depth_map_height / original_image_height
    scaled_intrinsics = scale_intrinsic(camera_intrinsics, scale_x, scale_y)

    # Apply metric scale to the depth map based on the model name
    if model_name == "DA3METRIC-LARGE":
        focal = (scaled_intrinsics[0, 0] + scaled_intrinsics[1, 1]) / 2.0
        metric_depth = focal * depth_map / 300.0
    else:
        metric_depth = depth_map.copy()  # No scaling applied for other models

    # Apply morphological operations to the mask if specified
    if mask is not None and morph_kernel_sizes is not None:
        mask = mask_morphology(mask, kernel_sizes=morph_kernel_sizes,
                                ratio_morphology=ratio_morphology)
    # Unproject the depth_map to 3D points in camera coordinates
    points = depth_map_to_point_cloud(metric_depth, scaled_intrinsics, mask)

    return points
