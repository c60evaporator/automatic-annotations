"""Depth-Anything-3 のラッパー（公式リポジトリ版）.

重みは HuggingFace から自動ダウンロードされる。
compose が ~/.cache/huggingface をマウントしているので、
ホスト側とキャッシュを共有し、再ダウンロードは起きない。

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
    *,
    camera_intrinsics: np.ndarray | list | None = None,
    original_image_width: int | None = None,
    original_image_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """推論結果をメトリック深度と、深度解像度に合わせた内部パラメータへ変換する.

    Returns:
        (metric_depths ``(N, H, W)``, scaled_intrinsics ``(N, 3, 3)``)

    NOTE: 内部パラメータは**深度マップの解像度**に合わせてスケールする。
    元画像の内部パラメータをそのまま使うと、点群が実際より広がる。
    """
    if model_name in METRIC_FROM_INTRINSIC_MODELS:
        if (camera_intrinsics is None or original_image_width is None
                or original_image_height is None):
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

        depth_height, depth_width = predicted[0].shape
        scale_x = depth_width / original_image_width
        scale_y = depth_height / original_image_height

        scaled_list, metric_list = [], []
        for index in range(num_preds):
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


class DepthAnything3Estimator:
    """DA3 による単眼深度推定."""

    def __init__(self, model_name: str = "DA3METRIC-LARGE", device: str = "cuda") -> None:
        # 重い import はここで行う。モジュールのトップに置くと
        # サーバー起動時に torch が読み込まれ、1 つでも壊れていると
        # サーバー自体が起動しなくなる
        from depth_anything_3.api import DepthAnything3

        self.model_name = model_name
        self.device = device
        logger.info("loading Depth-Anything-3: %s", model_name)
        self.model = DepthAnything3.from_pretrained(
            f"depth-anything/{model_name}"
        ).to(device=device)
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
        from depth_anything_3.utils.alignment import compute_sky_mask

        prediction = self.model.inference([image])

        if self.model_name in METRIC_NATIVE_MODELS:
            depths, intrinsics = get_metric_depth(prediction, self.model_name)
        else:
            depths, intrinsics = get_metric_depth(
                prediction, self.model_name,
                camera_intrinsics=[camera_intrinsic],
                original_image_width=image.width,
                original_image_height=image.height,
            )

        non_sky_mask = None
        if getattr(prediction, "sky", None) is not None:
            # compute_sky_mask は「空でない領域」を True で返す
            non_sky_mask = np.asarray(compute_sky_mask(prediction.sky[0]), dtype=bool)

        return {
            "metric_depth": np.asarray(depths[0], dtype=np.float32),
            "scaled_intrinsic": np.asarray(intrinsics[0], dtype=np.float64),
            "non_sky_mask": non_sky_mask,
        }
