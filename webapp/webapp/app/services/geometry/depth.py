import numpy as np

from ..schemas import Instance2D

from ..geometry.transform import (
    make_transform,
    transform_points,
)
from ..geometry.crop_resize import paste_cropped_mask
from ..visualize.segmentation import get_rgb_color


def transform_cam_to_ego(
    points_camera: np.ndarray,
    camera_translation: np.ndarray | list[float],
    camera_quaternion: np.ndarray | list[float],
) -> np.ndarray:
    """Transform a point cloud in camera coordinates into ego coordinates. (Intended for pseudo-LiDAR point clouds)

    Args:
        points_camera:
            shape=(N, 3) point cloud in camera coordinates.
        camera_translation:
            Ego coordinates of the camera origin [x, y, z].
        camera_quaternion:
            Rotation from the camera coordinates to the ego coordinates [w, x, y, z].

    Returns:
        shape=(N, 3) points in the ego coordinate system.
    """
    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(
            "points_camera must have shape (N, 3), "
            f"got {points_camera.shape}"
        )

    # Apply the camera -> ego pose represented by the calibration as it is.
    camera_to_ego = make_transform(camera_quaternion, camera_translation)

    return transform_points(points_camera, camera_to_ego)


def depth_map_to_point_cloud(
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
    depth_threshold: float = None,
) -> np.ndarray | list[np.ndarray]:
    """
    Generate a pseudo-LiDAR point cloud from a depth map and camera intrinsics filtered by a mask or a list of masks.

    Args:
        depth_map (np.ndarray): HxW array of depth values.
        camera_intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        depth_threshold (float, optional): If provided, points with depth greater than this threshold will be filtered out. Defaults to None.

    Returns:
        np.ndarray: Nx3 array of 3D points in the camera coordinate system
    """
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    i = i.flatten()
    j = j.flatten()
    depth = depth_map.flatten()

    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    x = (i - cx) * depth / fx
    y = (j - cy) * depth / fy
    z = depth

    points = np.vstack((x, y, z)).T

    # Filter points based on depth threshold if provided
    if depth_threshold is not None:
        points = points[points[:, 2] <= depth_threshold]

    return points


def depth_map_to_masked_point_clouds(
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
    masks: list[np.ndarray],
    common_mask: np.ndarray | None = None,
    depth_threshold: float = None,
) -> list[np.ndarray]:
    """
    Generate a pseudo-LiDAR point cloud from a depth map and camera intrinsics filtered by a mask or a list of masks.

    Args:
        depth_map (np.ndarray): HxW array of depth values.
        camera_intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        masks (list[np.ndarray] | None): List of HxW binary mask arrays. Only points where the masks are True will be included.
        common_mask (np.ndarray | None): HxW binary mask array for common areas to be included such as non-sky and non-ground.
        depth_threshold (float, optional): If provided, points with depth greater than this threshold will be filtered out. Defaults to None.

    Returns:
        list[np.ndarray]: List of Nx3 arrays of 3D points for each mask in masks in the camera coordinate system.
    """
    # Apply the common included mask to the mask / masks if provided
    if common_mask is not None:
        masks = [np.logical_and(m, common_mask) for m in masks]

    points = depth_map_to_point_cloud(depth_map, camera_intrinsics, depth_threshold=depth_threshold)
    masked_points = [points[m.flatten()] for m in masks]

    return masked_points


def depth_map_to_point_cloud_per_instance(metric_depth: np.ndarray,
                                          camera_intrinsics: np.ndarray,
                                          instances: list[Instance2D],
                                          common_mask: np.ndarray | None = None,
                                          depth_threshold: float | None = None,
                                          color: str | tuple[float, float, float] | dict[str, str] | dict[tuple[float, float, float], str] = "red",
                                          color_attr: str = "label",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generate a list of pseudo-LiDAR point clouds from a depth map, camera intrinsics, and a list of binary masks.

    Args:
        metric_depth (np.ndarray): HxW array of metric depth values.
        camera_intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        instances (list[Instance2D]): List of Instance2D objects that contain instance masks and their corresponding regions.
        common_mask (np.ndarray | None): HxW binary mask array for common areas to be included. Only points where common_mask is True will be included.
        depth_threshold (float | None): If provided, points with depth greater than this threshold will be filtered out. Defaults to None.
        color: Color used to render the point cloud. If a dict is given, the color is switched according to the ``instances.box`` attribute specified by ``color_attr``.
        color_attr: Name of the ``instances.box`` attribute used when ``color`` is a dict. Either "label" or "track_id" can be specified.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing:
            - points (list[np.ndarray]): List of Nx3 arrays of 3D points for each instance in the camera coordinate system.
            - colors (list[np.ndarray]): List of Nx3 arrays of RGB colors in the [0, 1] range, one color per point.
              The color is common within an instance, and ``colors[i]`` has the same length as ``points[i]``
              so that ``np.concatenate(points)`` and ``np.concatenate(colors)`` can be passed to ``plot_pointcloud()``.
    """
    # Convert Instance2D masks to boolean full masks
    full_masks = [paste_cropped_mask(instance.mask,
                                     original_width=metric_depth.shape[1],
                                     original_height=metric_depth.shape[0],
                                     crop_xyxy=instance.mask_region)
                  for instance in instances]

    # Unproject the depth_map to 3D points in camera coordinates
    points = depth_map_to_masked_point_clouds(metric_depth, camera_intrinsics, full_masks,
                                              common_mask=common_mask,
                                              depth_threshold=depth_threshold)

    # Create RGB colors normalized to [0, 1] range, one color per instance
    if isinstance(color, dict):
        instance_colors = [np.asarray(get_rgb_color(color, instance, color_attr, allow_dict=True), dtype=np.float32)
                           for instance in instances]
    else:
        # The color is common to all the instances, so resolve it only once
        color_rgb = np.asarray(get_rgb_color(color, None, color_attr, allow_dict=True), dtype=np.float32)
        instance_colors = [color_rgb] * len(instances)

    # Repeat the instance color for every point so that colors and points have the same length
    colors = [np.tile(instance_color, (len(instance_points), 1))
              for instance_color, instance_points in zip(instance_colors, points)]

    return points, colors
