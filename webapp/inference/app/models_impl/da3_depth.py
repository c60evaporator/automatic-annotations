"""Depth-Anything-3 のラッパー（公式リポジトリ版）.

重みは HuggingFace から自動ダウンロードされる。
compose がホストの ~/.cache/huggingface を HF_HOME にマウントしているので、
ホスト側とキャッシュを共有し、再ダウンロードは起きない。

推論は models/depth_anything3/src/inference.py と同じ関数構成にしてある:
  - predict_metric_depth: 姿勢なしの推論（メトリック深度・内部パラメータ・空マスク）
  - predict_pose_conditioned_depth: 内部パラメータと姿勢を与えた複数フレーム推論
  - get_metric_depth: 推論結果をメトリック深度と深度解像度の内部パラメータへ変換

torch / depth_anything_3 への import はこのモジュール内に閉じてあり、
モデルの初回ロード時にだけ読み込まれる。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from app.core.logging import get_logger
from common.transform3d import scale_intrinsic

logger = get_logger(__name__)

# メトリック深度を直接出す（内部パラメータからの換算が要る）モデル
METRIC_FROM_INTRINSIC_MODELS = {"DA3METRIC-LARGE"}
# 既にメトリックで、内部パラメータも自前で返すモデル
METRIC_NATIVE_MODELS = {"DA3NESTED-GIANT-LARGE-1.1", "DA3NESTED-GIANT-LARGE"}

# DA3METRIC-LARGE の換算に使う定数（参考実装に合わせる）
METRIC_FOCAL_DIVISOR = 300.0


def get_metric_depth(
    prediction: Any,
    model_name: str,
    camera_intrinsics: np.ndarray | list | None = None,
    original_image_widths: list[int] | None = None,
    original_image_heights: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """推論結果をメトリック深度と、深度解像度に合わせた内部パラメータへ変換する.

    Args:
        model_name: DA3 のモデル名（大文字・小文字は区別しない）
        camera_intrinsics: 元画像の内部パラメータ ``(N, 3, 3)``。DA3METRIC-LARGE で必須
        original_image_widths / original_image_heights: 元画像のサイズ。DA3METRIC-LARGE で必須

    Returns:
        (metric_depths ``(N, H, W)``, scaled_intrinsics ``(N, 3, 3)``)

    NOTE: 内部パラメータは**深度マップの解像度**に合わせてスケールする。
    元画像の内部パラメータをそのまま使うと、点群が実際より広がる。
    """
    model_name = model_name.upper()
    if model_name in METRIC_FROM_INTRINSIC_MODELS:
        if (camera_intrinsics is None or original_image_widths is None
                or original_image_heights is None):
            raise ValueError(
                f"{model_name} には camera_intrinsics と元画像サイズが必要です"
            )
        predicted = prediction.depth
        num_preds = predicted.shape[0]

        intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
        if intrinsics.shape[0] != num_preds:
            raise ValueError(
                f"camera_intrinsics の数 ({intrinsics.shape[0]}) と"
                f"推論結果の数 ({num_preds}) が一致しません"
            )

        scaled_list, metric_list = [], []
        for index in range(num_preds):
            depth_height, depth_width = predicted[index].shape
            scale_x = depth_width / original_image_widths[index]
            scale_y = depth_height / original_image_heights[index]
            scaled = scale_intrinsic(intrinsics[index], scale_x, scale_y)
            focal = (scaled[0, 0] + scaled[1, 1]) / 2.0
            metric_list.append(focal * predicted[index] / METRIC_FOCAL_DIVISOR)
            scaled_list.append(scaled)
        return np.stack(metric_list), np.stack(scaled_list)

    if model_name in METRIC_NATIVE_MODELS:
        if camera_intrinsics is not None:
            raise ValueError(
                f"{model_name} は内部パラメータを自前で返すため、"
                "camera_intrinsics を渡さないでください"
            )
        return prediction.depth, prediction.intrinsics

    raise ValueError(
        f"メトリック深度に対応していないモデルです: {model_name}. "
        f"対応: {sorted(METRIC_FROM_INTRINSIC_MODELS | METRIC_NATIVE_MODELS)}"
    )


def predict_metric_depth(
    model: Any,
    model_name: str,
    images: list[Image.Image],
    camera_intrinsics: np.ndarray | list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """DA3 で姿勢なしの深度推定を行い、メトリック深度に変換する.

    Args:
        model: DepthAnything3 のモデル
        model_name: DA3 のモデル名（get_metric_depth の換算方法を決める）
        images: 入力画像
        camera_intrinsics: 元画像の内部パラメータ ``(N, 3, 3)``。
            DA3METRIC-LARGE で必須、DA3NESTED 系では None

    Returns:
        (metric_depths ``(N, H, W)``, scaled_intrinsics ``(N, 3, 3)``,
         non_sky_masks ``(N, H, W)`` の bool。モデルが空を返さない場合は None)
    """
    from depth_anything_3.utils.alignment import compute_sky_mask

    prediction = model.inference(images)
    metric_depths, scaled_intrinsics = get_metric_depth(
        prediction,
        model_name=model_name,
        camera_intrinsics=camera_intrinsics,
        original_image_widths=[image.width for image in images],
        original_image_heights=[image.height for image in images],
    )
    non_sky_masks = None
    if getattr(prediction, "sky", None) is not None:
        # compute_sky_mask は「空でない領域」を True で返す
        non_sky_masks = np.asarray(compute_sky_mask(prediction.sky), dtype=bool)
    return metric_depths, scaled_intrinsics, non_sky_masks


def predict_pose_conditioned_depth(
    model: Any,
    model_name: str,
    images: list[Image.Image],
    camera_intrinsics: list[Any],
    camera_poses: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """DA3 で内部パラメータと姿勢を与えた複数フレーム深度推定を行う.

    Args:
        camera_intrinsics: 元画像の内部パラメータ（各 3x3）
        camera_poses: world→camera の外部パラメータ（各 4x4、OpenCV 座標系）

    Returns:
        predict_metric_depth と同じ
    """
    from depth_anything_3.utils.alignment import compute_sky_mask

    prediction = model.inference(
        images,
        extrinsics=np.stack(camera_poses, axis=0).astype(np.float32),
        intrinsics=np.stack(camera_intrinsics, axis=0).astype(np.float32),
    )
    metric_depths, scaled_intrinsics = get_metric_depth(prediction, model_name=model_name)
    non_sky_masks = None
    if getattr(prediction, "sky", None) is not None:
        non_sky_masks = np.asarray(compute_sky_mask(prediction.sky), dtype=bool)
    return metric_depths, scaled_intrinsics, non_sky_masks


class DepthAnything3Estimator:
    """DA3 による深度推定.

    DA3NESTED 系は姿勢付きの複数フレーム推論（estimate_window）、
    DA3METRIC-LARGE は単眼推論（estimate）で使う。
    """

    def __init__(self, model_id: str = "depth-anything/da3metric-large", device: str = "cuda") -> None:
        # 重い import はここで行う。モジュールのトップに置くと
        # サーバー起動時に torch が読み込まれ、1 つでも壊れていると
        # サーバー自体が起動しなくなる
        from depth_anything_3.api import DepthAnything3

        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]
        # 内部パラメータを自前で返すモデルは、姿勢を与えて複数フレームで推論する
        self.pose_conditioned = self.model_name.upper() in METRIC_NATIVE_MODELS
        self.device = device
        logger.info("loading Depth-Anything-3: %s", model_id)
        self.model = DepthAnything3.from_pretrained(model_id).to(device=device)
        logger.info("Depth-Anything-3 ready on %s", device)

    def to(self, device: str):
        """models.py の解放処理から呼ばれる（VRAM を返すため）."""
        self.model.to(device=device)
        self.device = device
        return self

    def estimate(
        self, image: Image.Image, camera_intrinsic: Any
    ) -> dict[str, Any]:
        """1 枚の画像から、メトリック深度・内部パラメータ・空マスクを得る.

        Returns:
            {"metric_depth", "scaled_intrinsic", "non_sky_mask"}
            non_sky_mask はモデルが空を返さない場合 None。
        """
        depths, intrinsics, non_sky_masks = predict_metric_depth(
            self.model, self.model_name, [image], camera_intrinsics=[camera_intrinsic],
        )
        return {
            "metric_depth": np.asarray(depths[0], dtype=np.float32),
            "scaled_intrinsic": np.asarray(intrinsics[0], dtype=np.float64),
            "non_sky_mask": None if non_sky_masks is None else non_sky_masks[0],
        }

    def estimate_window(
        self,
        images: list[Image.Image],
        camera_intrinsics: list[Any],
        camera_poses: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """1 ウィンドウの深度を推定する（戻り値は predict_pose_conditioned_depth と同じ）."""
        return predict_pose_conditioned_depth(
            self.model, self.model_name, images, camera_intrinsics, camera_poses
        )
