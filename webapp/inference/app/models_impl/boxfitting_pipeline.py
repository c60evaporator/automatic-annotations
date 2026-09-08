"""Depth Estimation & Box Fitting の本実装.

スタブと同じインターフェースを持つ:
    estimate_depth(frame, output_dir, ...)          … カメラフレームごと
    load_lidar(frame, output_dir, ...)              … sample ごと
    process_frame_instances(instances, depth_result, ...) … フレーム単位

インスタンス処理をフレーム単位にしているのは、深度マップと空マスクを
1 フレームにつき 1 回だけ読めば済むようにするため。
インスタンスごとに読み直すと、同じ .npz を何度も展開することになる。

現時点の制約:
  - LiDAR を使った混合は未実装（use_lidar=True でも深度点群のみ）
  - Box Fitting のアルゴリズムは未実装（点群までを保存する）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.logging import get_logger
from app.services.depth_ops import (
    close_mask,
    depth_map_to_point_cloud,
    instance_points_from_depth,
    resize_mask_nearest,
    rle_to_depth_mask,
)
from common.mask_rle import encode_rle
from common.point_ops import downsample_to_max, points_to_json
from common.transform3d import camera_to_ego

logger = get_logger(__name__)

# 点群がこの数に満たないインスタンスは「点が少なすぎる」として扱う
MIN_POINTS_FOR_BOX = 10


class BoxFittingPipeline:
    """DA3 による深度推定と、インスタンスごとの点群生成."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        from app.models_impl.da3_depth import DepthAnything3Estimator

        self.estimator = DepthAnything3Estimator(model_name, device=device)
        self.device = device

    def to(self, device: str):
        self.estimator.to(device)
        self.device = device
        return self

    # ── 1. 深度推定 ───────────────────────────────────────────────────────

    def estimate_depth(
        self,
        frame: dict[str, Any],
        output_dir: Path,
        *,
        dataroot: Path | str = "",
        downscale: float = 1.0,
        **_: Any,
    ) -> dict[str, Any]:
        """1 カメラフレームの深度を推定し、.npz として保存する.

        保存内容:
            depth      … メトリック深度 (H, W) float16
            intrinsic  … **その解像度に対する** 内部パラメータ
            non_sky    … 空でない領域のマスク（モデルが返す場合のみ）

        内部パラメータを一緒に保存するのは、後段（点群化）で
        「保存した解像度に合う K」が必要だから。別々に持つと、
        倍率の食い違いに気づけないまま点群がずれる。
        """
        calib = frame.get("calibrated_sensor") or {}
        image_path = Path(dataroot) / frame["filename"]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        result = self.estimator.estimate(image, calib.get("camera_intrinsic"))
        depth = result["metric_depth"]
        intrinsic = result["scaled_intrinsic"]
        non_sky = result["non_sky_mask"]

        # 容量削減のための追加縮小。内部パラメータも同じ倍率で合わせる
        if 0 < downscale < 1.0:
            height = max(1, int(depth.shape[0] * downscale))
            width = max(1, int(depth.shape[1] * downscale))
            depth, intrinsic, non_sky = _resize_depth(
                depth, intrinsic, non_sky, height, width
            )

        path = output_dir / "depth" / f"{frame['sample_data_token']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"depth": depth.astype(np.float16), "intrinsic": intrinsic}
        if non_sky is not None:
            arrays["non_sky"] = non_sky
        np.savez_compressed(path, **arrays)

        valid = depth[np.isfinite(depth) & (depth > 0)]
        return {
            "sample_data_token": frame["sample_data_token"],
            # DERIVED_ROOT からの相対パスで返す（絶対パスを DB に入れない）
            "depth_path": str(path.relative_to(output_dir.parent.parent)),
            "depth_width": int(depth.shape[1]),
            "depth_height": int(depth.shape[0]),
            "scale": 1.0,
            "shift": 0.0,
            "min_depth": float(valid.min()) if valid.size else None,
            "max_depth": float(valid.max()) if valid.size else None,
            "num_points": int(valid.size),
        }

    # ── 2. LiDAR ──────────────────────────────────────────────────────────

    def load_lidar(
        self,
        frame: dict[str, Any],
        output_dir: Path,
        *,
        dataroot: Path | str = "",
        sweeps: list[dict[str, Any]] | None = None,
        num_sweeps: int = 1,
        **_: Any,
    ) -> dict[str, Any]:
        """sweep を統合し、地面判定を付けて .npz として保存する.

        Args:
            frame: キーフレームの LiDAR sample_data（座標系の基準）
            sweeps: 統合対象のフレーム（古い順、末尾がキーフレーム）。
                省略時はキーフレームのみ

        use_lidar に関係なく常に呼ぶ。UI が比較用に生 LiDAR と
        地面を表示できるようにするため。

        保存は **ego 座標**。深度由来の点群と同じ座標系に揃えておかないと、
        UI で重ねたときに位置が合わない。
        """
        from app.services.lidar_ops import (
            lidar_to_ego,
            merge_sweeps_with_ground,
        )

        target_sweeps = sweeps or [frame]
        calib = frame["calibrated_sensor"]
        # Patchwork++ は sensor_height を基準に地面を探すので、
        # 実際の取り付け高さを渡す（既定値のままだと車種差で外れる）
        sensor_height = float(calib["translation"][2])

        points_lidar, ground_mask = merge_sweeps_with_ground(
            target_sweeps, Path(dataroot), sensor_height=sensor_height
        )
        points_ego = lidar_to_ego(points_lidar, calib)

        path = output_dir / "lidar" / f"{frame['sample_token']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points=points_ego.astype(np.float32),
            ground_mask=ground_mask,
        )

        return {
            "sample_token": frame["sample_token"],
            "sample_data_token": frame["sample_data_token"],
            "pointcloud_path": str(path.relative_to(output_dir.parent.parent)),
            "coordinate_frame": "ego",
            "num_sweeps": len(target_sweeps),
            "num_points": int(points_ego.shape[0]),
            "num_ground_points": int(ground_mask.sum()),
        }

    # ── 3. インスタンスごとの点群 ────────────────────────────────────────

    def process_frame_instances(
        self,
        instances: list[dict[str, Any]],
        depth_result: dict[str, Any],
        output_dir: Path,
        *,
        frame: dict[str, Any],
        mask_params: dict[str, Any],
        depth_params: dict[str, Any],
        use_lidar: bool = False,
        stored_points_max: int = 500,
        max_depth: float | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """1 フレーム分のインスタンスを処理する.

        深度マップと空マスクはここで 1 回だけ読み、
        全インスタンスで使い回す。
        """
        results: list[dict[str, Any]] = []
        if not instances:
            return results

        path = output_dir.parent.parent / depth_result["depth_path"]
        if not path.exists():
            logger.warning("depth map not found: %s", path)
            return [_failed(inst, "no_points") for inst in instances]

        with np.load(path) as data:
            depth = np.asarray(data["depth"], dtype=np.float32)
            intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
            non_sky = (
                np.asarray(data["non_sky"], dtype=bool)
                if "non_sky" in data.files else None
            )

        depth_height, depth_width = depth.shape
        if non_sky is not None and non_sky.shape != depth.shape:
            non_sky = resize_mask_nearest(non_sky, depth_height, depth_width)

        # 全画素の点群は 1 回だけ作り、マスクで切り出す
        all_points = depth_map_to_point_cloud(
            depth, intrinsic, depth_threshold=max_depth
        )

        calib = frame.get("calibrated_sensor") or {}
        dilation = int(mask_params.get("dilation", 1))
        erosion = int(mask_params.get("erosion", 1))

        for instance in instances:
            mask = rle_to_depth_mask(
                instance["mask_rle"], depth_height, depth_width
            )
            closed = close_mask(mask, dilation, erosion)

            points_camera, points_raw_camera = instance_points_from_depth(
                all_points, closed,
                common_mask=non_sky,
                ror_nb_points=int(depth_params.get("ror_nb_points", 0)),
                ror_radius=float(depth_params.get("ror_radius", 0.0)),
                dbscan_eps=float(depth_params.get("dbscan_eps", 0.0)),
                dbscan_min_samples=int(depth_params.get("dbscan_min_samples", 0)),
            )
            raw_count = int(points_raw_camera.shape[0])

            base = {
                "instance_tracking_2d_id": instance["instance_tracking_2d_id"],
                "sample_data_token": instance["sample_data_token"],
                "track_id": str(instance["track_id"]),
                "label": instance["label"],
                # クロージング後のマスクは webapp で再計算できないので保存する
                "mask_rle_closed": encode_rle(closed),
                "num_points_depth": raw_count,
                "num_points_lidar": 0,
                "num_points_depth_kept": int(points_camera.shape[0]),
                "num_points_lidar_kept": 0,
            }

            if raw_count == 0:
                results.append({**base, "status": "no_points"})
                continue
            if points_camera.shape[0] < MIN_POINTS_FOR_BOX:
                results.append({**base, "status": "too_few_points"})
                continue

            points_ego = camera_to_ego(
                points_camera, calib["translation"], calib["rotation"]
            )
            results.append({
                **base,
                # Box Fitting は未実装。点群までを保存し、
                # ボックスが無いことが status で分かるようにする
                "status": "not_fitted",
                # フィルタ前は保存しない。深度マップとクロージング後マスクから
                # 再生成できるため（再フィルタ用エンドポイント経由）
                "points_depth_ego": points_to_json(
                    downsample_to_max(points_ego, stored_points_max)
                ),
                "points_lidar_ego": None,
            })

        return results


def _failed(instance: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "instance_tracking_2d_id": instance["instance_tracking_2d_id"],
        "sample_data_token": instance["sample_data_token"],
        "track_id": str(instance["track_id"]),
        "label": instance["label"],
        "status": status,
        "num_points_depth": 0,
        "num_points_lidar": 0,
    }


def _resize_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    non_sky: np.ndarray | None,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """深度・内部パラメータ・空マスクをまとめて縮小する.

    深度は最近傍で間引く。補間すると、物体の縁で手前と奥の中間値が
    生まれ、実在しない位置に点が出る。
    """
    from common.transform3d import scale_intrinsic

    src_height, src_width = depth.shape
    rows = ((np.arange(height) + 0.5) * src_height / height).astype(np.int64)
    cols = ((np.arange(width) + 0.5) * src_width / width).astype(np.int64)
    rows = rows.clip(0, src_height - 1)
    cols = cols.clip(0, src_width - 1)

    resized = depth[rows][:, cols]
    scaled = scale_intrinsic(intrinsic, width / src_width, height / src_height)
    resized_sky = non_sky[rows][:, cols] if non_sky is not None else None
    return resized, scaled, resized_sky
