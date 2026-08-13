import numpy as np

from ..common.geometry.crop_resize import scale_intrinsic
from ..common.geometry.depth import depth_map_to_point_cloud
from ..common.image_processing.segmentation import mask_morphology

from depth_anything_3.specs import Prediction

def get_metric_depth(prediction: Prediction,
                     model_name: str,
                     camera_intrinsics: np.ndarray | list | None = None,
                     original_image_width: int | None = None,
                     original_image_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the depth prediction to metric depth using the camera intrinsics and original image size.

    Only supported for the following models:
    - 'DA3METRIC-LARGE': The depth is predicted in a normalized scale and needs to be converted to metric depth using the camera intrinsics and original image size.
    - 'DA3NESTED-GIANT-LARGE-1.1' and 'DA3NESTED-GIANT-LARGE': The depth is already in metric scale and can be returned directly.

    Args:
        prediction (Prediction): The depth prediction from the model.
        model_name (str): The name of the model used for inference.
        camera_intrinsics (np.ndarray | list | None): The camera intrinsic matrix (3x3) or a list of intrinsic matrices for each image in the batch. Required for 'DA3METRIC-LARGE' model.
        original_image_width (int | None): The width of the original image before any resizing. Required for 'DA3METRIC-LARGE' model.
        original_image_height (int | None): The height of the original image before any resizing. Required for 'DA3METRIC-LARGE' model.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - metric_depth (np.ndarray): The depth map in metric scale.
            - scaled_intrinsics (np.ndarray): The scaled camera intrinsic matrix (3x3)
    """
    # Calculate the intrinsic if the model doesn't provide pose estimation
    if model_name == "DA3METRIC-LARGE":
        if camera_intrinsics is None or original_image_width is None or original_image_height is None:
            raise ValueError("For model_name 'DA3METRIC-LARGE', both `camera_intrinsics`, `original_image_width` and `original_image_height` must be provided.")

        predicted_depths = prediction.depth
        num_preds = predicted_depths.shape[0]

        # Convert camera_intrinsics to a numpy array if it's a list and check the number of predictions
        if isinstance(camera_intrinsics, list):
            camera_intrinsics = np.array(camera_intrinsics)
        if camera_intrinsics.shape[0] != num_preds:
            raise ValueError(f"The number of camera_intrinsics ({camera_intrinsics.shape[0]}) does not match the number of predicted depths ({num_preds}).")

        # Get the scale factors for the depth map and original image
        depth_map_height, depth_map_width = prediction.depth[0].shape
        scale_x = depth_map_width / original_image_width
        scale_y = depth_map_height / original_image_height

        # Calculate the metric depth
        scaled_intrinsics = []
        metric_depths = []
        for i in range(num_preds):
            scaled_intrinsic = scale_intrinsic(camera_intrinsics[i], scale_x, scale_y)
            focal = (scaled_intrinsic[0, 0] + scaled_intrinsic[1, 1]) / 2.0
            metric_depth = focal * predicted_depths[i] / 300.0
            scaled_intrinsics.append(scaled_intrinsic)
            metric_depths.append(metric_depth)
        
        # Stack the metric depths and scaled intrinsics into numpy arrays
        metric_depths = np.stack(metric_depths, axis=0)
        scaled_intrinsics = np.stack(scaled_intrinsics, axis=0)

        return metric_depths, scaled_intrinsics

    # For these models, the depth is already in metric scale
    elif model_name in ["DA3NESTED-GIANT-LARGE-1.1", "DA3NESTED-GIANT-LARGE"]:
        if camera_intrinsics is not None or original_image_width is not None or original_image_height is not None:
            raise ValueError("For model_name 'DA3NESTED-GIANT-LARGE-1.1' and 'DA3NESTED-GIANT-LARGE', `camera_intrinsics`, `original_image_width` and `original_image_height` must not be provided.")
        return prediction.depth, prediction.intrinsics

    else:
        raise ValueError(f"Metric output is not supported. model_name: {model_name}. Supported models are 'DA3METRIC-LARGE', 'DA3NESTED-GIANT-LARGE-1.1', and 'DA3NESTED-GIANT-LARGE'.")


def get_pseudo_lidar(metric_depth: np.ndarray,
                     scaled_intrinsics: np.ndarray,
                     sky_mask: np.ndarray | None = None,
                     mask: np.ndarray | None = None,
                     masks: list[np.ndarray] | None = None,
                     morph_kernel_sizes: list[int] | None = None,
                     ratio_morphology: bool = False,
) -> np.ndarray:
    """
    Generate a pseudo-LiDAR point cloud from a depth map and camera intrinsics.

    Args:
        metric_depth (np.ndarray): HxW array of metric depth values.
        scaled_intrinsics (np.ndarray): 3x3 scaled camera intrinsic matrix.
        sky_mask (np.ndarray | None): HxW binary mask array for the sky. Points where sky_mask is True will be excluded.
        mask (np.ndarray | None): HxW binary mask array. Only points where mask is True will be included. If None, all points are included.
        masks (list[np.ndarray] | None): List of HxW binary mask arrays. Only points where the masks are True will be included. If None, all points are included.
        morph_kernel_sizes (list[int] | None): List of kernel sizes for morphological operations. Positive size represents dilation, whereas negative size represents erosion. If None, no morphological operations are applied.
        ratio_morphology (bool): Whether to apply ratio-based morphology. If True, the applied kernel sizes are computed by multiplying `morph_kernel_sizes` by the average of the mask bounding box height and width.

    Returns:
        np.ndarray: Nx3 array of 3D points in the camera coordinate system. Only points where mask is True are included.
    """
    if masks is not None and mask is not None:
        raise ValueError("Only one of `mask` or `masks` can be provided, not both.")

    # Apply morphological operations to the mask if specified
    if morph_kernel_sizes is not None:
        if mask is not None:
            mask = mask_morphology(mask, kernel_sizes=morph_kernel_sizes,
                                ratio_morphology=ratio_morphology)
        elif masks is not None:
            masks = [mask_morphology(m, kernel_sizes=morph_kernel_sizes,
                                ratio_morphology=ratio_morphology) for m in masks]

    # Apply the sky mask to the mask / masks if provided
    if sky_mask is not None:
        if mask is not None:
            mask = np.logical_and(mask, np.logical_not(sky_mask))
        elif masks is not None:
            masks = [np.logical_and(m, np.logical_not(sky_mask)) for m in masks]

    # Unproject the depth_map to 3D points in camera coordinates
    points = depth_map_to_point_cloud(metric_depth, scaled_intrinsics, mask, masks)

    return points
