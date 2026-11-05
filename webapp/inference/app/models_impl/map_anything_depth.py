"""MapAnything のラッパー（公式リポジトリ版）.

重みは HuggingFace から HF_HOME へ自動ダウンロードされる。
カメラの内部パラメータと姿勢を入力に与えた複数フレーム推論のみを行う。

torch / mapanything への import はこのモジュール内に閉じてあり、
モデルの初回ロード時にだけ読み込まれる。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from app.core.logging import get_logger
from common.transform3d import scale_intrinsic

logger = get_logger(__name__)


def _resample_to_resized_view(
    depth: np.ndarray,
    mask: np.ndarray,
    processed_intrinsic: np.ndarray,
    original_intrinsic: np.ndarray,
    original_width: int,
    original_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """中央切り出しされた深度を、元画像を単純リサイズした画素格子へ移す.

    MapAnything の前処理はアスペクト比を保ってリサイズした後に中央を切り出す。
    後段はインスタンスマスクを単純リサイズで深度の解像度へ合わせるので、
    そのままでは画像の端ほどマスクと深度の位置がずれる。
    同じ視線の画素を最近傍で引くため、z 深度の値は変わらない。
    切り出しで失われた範囲は無効（mask=False, depth=0）とする。

    Returns:
        (depth, mask, 単純リサイズに対応する内部パラメータ)
    """
    height, width = depth.shape
    intrinsic = scale_intrinsic(
        original_intrinsic, width / original_width, height / original_height
    )
    target_u = np.arange(width, dtype=np.float64)
    target_v = np.arange(height, dtype=np.float64)
    source_u = np.rint(
        (target_u - intrinsic[0, 2]) / intrinsic[0, 0] * processed_intrinsic[0, 0]
        + processed_intrinsic[0, 2]
    ).astype(np.int64)
    source_v = np.rint(
        (target_v - intrinsic[1, 2]) / intrinsic[1, 1] * processed_intrinsic[1, 1]
        + processed_intrinsic[1, 2]
    ).astype(np.int64)
    valid_u = (source_u >= 0) & (source_u < width)
    valid_v = (source_v >= 0) & (source_v < height)

    rows = np.clip(source_v, 0, height - 1)[:, None]
    cols = np.clip(source_u, 0, width - 1)[None, :]
    inside = valid_v[:, None] & valid_u[None, :]
    return (
        np.where(inside, depth[rows, cols], 0.0).astype(depth.dtype),
        inside & mask[rows, cols],
        intrinsic,
    )


def predict_pose_conditioned_depth(
    model: Any,
    images: list[Image.Image],
    camera_intrinsics: list[np.ndarray],
    camera_poses: list[np.ndarray],
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MapAnything で内部パラメータと姿勢を与えた複数フレーム深度推定を行う.

    Args:
        model: MapAnything のモデル
        images: 入力画像
        camera_intrinsics: 元画像の内部パラメータ（各 3x3）
        camera_poses: world→camera の外部パラメータ（各 4x4、OpenCV 座標系、メートル単位）。
            Depth-Anything-3 と同じ形式で受け取り、camera→world へ変換して渡す
        device: 推論デバイス。bfloat16 の AMP は CUDA でのみ使う

    Returns:
        (metric_depths ``(N, H, W)``, scaled_intrinsics ``(N, 3, 3)``,
         有効領域のマスク ``(N, H, W)`` の bool)
    """
    import torch
    from mapanything.utils.image import preprocess_inputs

    cam_to_world = [np.linalg.inv(np.asarray(pose, dtype=np.float64)) for pose in camera_poses]
    # global 座標は桁が大きいので、原点を先頭カメラへ移す（カメラから見た深度は変わらない）
    origin = cam_to_world[0][:3, 3].copy()
    for pose in cam_to_world:
        pose[:3, 3] -= origin

    views = [
        {
            "img": torch.from_numpy(np.array(image.convert("RGB"))),
            "intrinsics": torch.from_numpy(np.asarray(intrinsic, dtype=np.float32)),
            "camera_poses": torch.from_numpy(pose.astype(np.float32)),
            "is_metric_scale": torch.tensor([True]),
        }
        for image, intrinsic, pose in zip(images, camera_intrinsics, cam_to_world, strict=True)
    ]
    processed_views = preprocess_inputs(views)
    predictions = model.infer(
        processed_views,
        memory_efficient_inference=True,
        use_amp=str(device).startswith("cuda"),
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )

    depths, intrinsics, masks = [], [], []
    for index, (processed_view, prediction) in enumerate(
        zip(processed_views, predictions, strict=True)
    ):
        # depth_z は model.infer() の中でメトリックスケールが適用済み
        depth, mask, intrinsic = _resample_to_resized_view(
            prediction["depth_z"][0, ..., 0].float().cpu().numpy(),
            prediction["mask"][0, ..., 0].cpu().numpy().astype(bool),
            processed_view["intrinsics"][0].float().cpu().numpy(),
            np.asarray(camera_intrinsics[index], dtype=np.float64),
            images[index].width,
            images[index].height,
        )
        depths.append(depth)
        masks.append(mask)
        intrinsics.append(intrinsic)
    return np.stack(depths), np.stack(intrinsics), np.stack(masks)


class MapAnythingEstimator:
    """MapAnything による姿勢付き複数フレーム深度推定."""

    pose_conditioned = True

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        # 重い import はここで行う（サーバー起動時に読み込まないため）
        from mapanything.models import MapAnything

        self.model_id = model_id
        self.device = device
        logger.info("loading MapAnything: %s", model_id)
        self.model = MapAnything.from_pretrained(model_id).to(device)
        logger.info("MapAnything ready on %s", device)

    def to(self, device: str):
        """解放処理から呼ばれる（VRAM を返すため）."""
        self.model.to(device)
        self.device = device
        return self

    def estimate_window(
        self,
        images: list[Image.Image],
        camera_intrinsics: list[Any],
        camera_poses: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1 ウィンドウの深度を推定する（戻り値は predict_pose_conditioned_depth と同じ）."""
        return predict_pose_conditioned_depth(
            self.model, images, camera_intrinsics, camera_poses, device=self.device
        )
