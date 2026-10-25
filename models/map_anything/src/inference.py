import numpy as np
import torch
from PIL import Image

from ..common.geometry.depth import depth_map_to_point_cloud, depth_map_to_masked_point_clouds

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

# Default options of MapAnything.infer() (see https://github.com/facebookresearch/map-anything#multi-modal-inference)
DEFAULT_INFER_KWARGS = {
    "memory_efficient_inference": True,
    "use_amp": True,
    "amp_dtype": "bf16",
    "apply_mask": True,
    "mask_edges": True,
}


def _run_inference(model: MapAnything,
                   views: list[dict],
                   infer_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the input views, run MapAnything and convert the outputs to numpy arrays.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - metric_depths (np.ndarray): (N, H, W) metric depth maps in the resolution of the processed images.
            - scaled_intrinsics (np.ndarray): (N, 3, 3) camera intrinsics corresponding to the depth maps.
              If the intrinsics are given as inputs, they are scaled to the processed images. Otherwise, the predicted ones are returned.
            - masks (np.ndarray): (N, H, W) bool masks of the valid depth pixels (e.g. sky and edge artifacts are False).
    """
    processed_views = preprocess_inputs(views)
    predictions = model.infer(processed_views, **{**DEFAULT_INFER_KWARGS, **(infer_kwargs or {})})

    metric_depths, scaled_intrinsics, masks = [], [], []
    for processed_view, pred in zip(processed_views, predictions):
        # depth_z is already in metric scale (the metric scaling factor is applied in model.infer())
        metric_depths.append(pred["depth_z"][0, ..., 0].float().cpu().numpy())
        intrinsics = processed_view["intrinsics"] if "intrinsics" in processed_view else pred["intrinsics"]
        scaled_intrinsics.append(intrinsics[0].float().cpu().numpy())
        masks.append(pred["mask"][0, ..., 0].cpu().numpy().astype(bool))

    return np.stack(metric_depths, axis=0), np.stack(scaled_intrinsics, axis=0), np.stack(masks, axis=0)


def _to_image_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image.convert("RGB")))  # (H, W, 3) uint8 [0, 255]


def predict_metric_depth(model: MapAnything,
                         images: list[Image.Image],
                         camera_intrinsics: np.ndarray | list | None = None,
                         infer_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict the metric depth of the images with MapAnything (without pose conditioning).

    Args:
        model (MapAnything): MapAnything model.
        images (list[Image.Image]): Input images. They are reconstructed jointly as multi views.
        camera_intrinsics (np.ndarray | list | None): The camera intrinsic matrices (3x3) of the original images.
            If provided, they are used as calibration inputs. Otherwise, they are predicted by the model.
        infer_kwargs (dict | None): Additional keyword arguments of MapAnything.infer(), which overwrite DEFAULT_INFER_KWARGS.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - metric_depths (np.ndarray): (N, H, W) metric depth maps.
            - scaled_intrinsics (np.ndarray): (N, 3, 3) camera intrinsics scaled to the depth maps.
            - masks (np.ndarray): (N, H, W) bool masks of the valid depth pixels.
    """
    if camera_intrinsics is not None and len(camera_intrinsics) != len(images):
        raise ValueError(f"The number of camera_intrinsics ({len(camera_intrinsics)}) does not match the number of images ({len(images)}).")

    views = []
    for i, image in enumerate(images):
        view = {"img": _to_image_tensor(image)}
        if camera_intrinsics is not None:
            view["intrinsics"] = torch.from_numpy(np.asarray(camera_intrinsics[i], dtype=np.float32))
        views.append(view)

    return _run_inference(model, views, infer_kwargs)


def predict_pose_conditioned_depth(model: MapAnything,
                                   images: list[Image.Image],
                                   camera_intrinsics: list[np.ndarray],
                                   camera_poses: list[np.ndarray],
                                   infer_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict the metric depth of the images with MapAnything conditioned on the camera intrinsics and poses.

    Args:
        model (MapAnything): MapAnything model.
        images (list[Image.Image]): Input images.
        camera_intrinsics (list[np.ndarray]): The camera intrinsic matrices (3x3) of the original images.
        camera_poses (list[np.ndarray]): The world-to-camera extrinsic matrices (4x4, OpenCV convention) in metric scale,
            same as the extrinsics of Depth-Anything-3. They are converted to the camera-to-world poses expected by MapAnything.
        infer_kwargs (dict | None): Additional keyword arguments of MapAnything.infer(), which overwrite DEFAULT_INFER_KWARGS.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - metric_depths (np.ndarray): (N, H, W) metric depth maps.
            - scaled_intrinsics (np.ndarray): (N, 3, 3) camera intrinsics scaled to the depth maps.
            - masks (np.ndarray): (N, H, W) bool masks of the valid depth pixels.
    """
    if not (len(images) == len(camera_intrinsics) == len(camera_poses)):
        raise ValueError(f"The numbers of images ({len(images)}), camera_intrinsics ({len(camera_intrinsics)}) "
                         f"and camera_poses ({len(camera_poses)}) must be the same.")

    # Convert the world-to-camera extrinsics to camera-to-world poses
    cam_to_world = [np.linalg.inv(np.asarray(pose, dtype=np.float64)) for pose in camera_poses]
    # Move the world origin to the first camera to avoid large coordinates (e.g. global coordinates of nuScenes).
    # This doesn't change the depth in the camera frames.
    origin = cam_to_world[0][:3, 3].copy()
    for pose in cam_to_world:
        pose[:3, 3] -= origin

    views = []
    for image, intrinsic, pose in zip(images, camera_intrinsics, cam_to_world):
        views.append({
            "img": _to_image_tensor(image),
            "intrinsics": torch.from_numpy(np.asarray(intrinsic, dtype=np.float32)),
            "camera_poses": torch.from_numpy(pose.astype(np.float32)),
            "is_metric_scale": torch.tensor([True]),
        })

    return _run_inference(model, views, infer_kwargs)


def get_pseudo_lidar(metric_depth: np.ndarray,
                     scaled_intrinsics: np.ndarray,
                     non_sky_mask: np.ndarray | None = None,
                     depth_threshold: float | None = None,
) -> np.ndarray:
    """
    Generate a pseudo-LiDAR point cloud from a depth map and camera intrinsics.

    Args:
        metric_depth (np.ndarray): HxW array of metric depth values.
        scaled_intrinsics (np.ndarray): 3x3 scaled camera intrinsic matrix.
        non_sky_mask (np.ndarray | None): HxW binary mask array for the valid areas. Only points where non_sky_mask is True will be included.
        depth_threshold (float | None): If provided, points with depth greater than this threshold will be filtered out. Defaults to None.

    Returns:
        np.ndarray: Nx3 array of 3D points in the camera coordinate system.
    """
    if non_sky_mask is not None:
        return depth_map_to_masked_point_clouds(metric_depth, scaled_intrinsics, [non_sky_mask],
                                                depth_threshold=depth_threshold)[0]

    else:
        return depth_map_to_point_cloud(metric_depth, scaled_intrinsics,
                                        depth_threshold=depth_threshold)
