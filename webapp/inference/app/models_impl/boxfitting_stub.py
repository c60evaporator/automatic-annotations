"""Depth Estimation & Box Fitting のスタブ.

UI とジョブの流れを確認するためのもので、実際の推論は入っていない。
本実装に差し替えるときは各メソッドのシグネチャを保つこと。

実物に近い形で **ファイルも実際に書く**（深度マップと LiDAR 点群の .npz）。
パスだけ返して中身が無いと、UI 側のファイル読み込み・削除まわりを
確認できないため。
"""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from common.mask_rle import decode_rle, encode_rle, rle_bbox
from common.point_ops import downsample_to_max, points_to_json

logger = get_logger(__name__)

DEFAULT_STUB_DELAY_SEC = 0.02

# スタブが生成する LiDAR 点数（1 sweep 相当）
STUB_LIDAR_POINTS_PER_SWEEP = 4_000


def _rng_for(*parts: Any) -> random.Random:
    """入力から決まる乱数。再実行しても同じ結果になるようにする."""
    seed = int(hashlib.md5("::".join(map(str, parts)).encode()).hexdigest()[:8], 16)
    return random.Random(seed)


class BoxFittingStub:
    def __init__(self, *_: Any, **__: Any) -> None:
        logger.warning(
            "Depth Estimation / Box Fitting はスタブで動作しています（推論は行いません）"
        )

    def to(self, device: str):
        return self

    # ── 1. 深度推定 ───────────────────────────────────────────────────────

    def estimate_depth(
        self,
        frame: dict[str, Any],
        output_dir: Path,
        *,
        downscale: float = 0.5,
        stub_delay_sec: float | None = None,
    ) -> dict[str, Any]:
        """1 カメラフレームの深度を推定し、.npz として保存する.

        保存解像度は元画像の downscale 倍。容量が 1/4 になる代わりに、
        読み出し側は内部パラメータを同じ倍率でスケールする必要がある
        （そのため depth_width / depth_height を結果に含める）。
        """
        time.sleep(DEFAULT_STUB_DELAY_SEC if stub_delay_sec is None else stub_delay_sec)

        width = int((frame.get("width") or 1600) * downscale)
        height = int((frame.get("height") or 900) * downscale)

        # 手前ほど近い、それらしい深度勾配を作る
        rows = np.linspace(60.0, 5.0, height, dtype=np.float32)[:, None]
        depth = np.repeat(rows, width, axis=1)
        depth += np.float32(_rng_for(frame["sample_data_token"]).uniform(-2, 2))
        depth = depth.astype(np.float16)

        path = output_dir / "depth" / f"{frame['sample_data_token']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, depth=depth)

        return {
            "sample_data_token": frame["sample_data_token"],
            # DERIVED_ROOT からの相対パスで返す（絶対パスを DB に入れない）
            "depth_path": str(path.relative_to(output_dir.parent.parent)),
            "depth_width": width,
            "depth_height": height,
            "scale": 1.0,
            "shift": 0.0,
            "min_depth": float(depth.min()),
            "max_depth": float(depth.max()),
            "num_points": int(width * height),
        }

    # ── 2. LiDAR の統合と地面除去 ────────────────────────────────────────

    def load_lidar(
        self,
        frame: dict[str, Any],
        output_dir: Path,
        *,
        num_sweeps: int = 1,
        stub_delay_sec: float | None = None,
    ) -> dict[str, Any]:
        """sweep を統合し、地面判定を付けて .npz として保存する.

        use_lidar に関係なく常に呼ぶ。UI が比較用に生 LiDAR と
        地面を表示できるようにするため。
        """
        time.sleep(DEFAULT_STUB_DELAY_SEC if stub_delay_sec is None else stub_delay_sec)

        rng = np.random.default_rng(
            int(hashlib.md5(frame["sample_token"].encode()).hexdigest()[:8], 16)
        )
        count = STUB_LIDAR_POINTS_PER_SWEEP * max(1, num_sweeps)
        points = np.column_stack([
            rng.uniform(-40, 60, count),
            rng.uniform(-30, 30, count),
            rng.normal(0.0, 1.2, count),
        ]).astype(np.float32)
        # z が低い点を地面とみなす（Patchwork++ の代わり）
        ground_mask = points[:, 2] < -0.8

        path = output_dir / "lidar" / f"{frame['sample_token']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, points=points, ground_mask=ground_mask)

        return {
            "sample_token": frame["sample_token"],
            "sample_data_token": frame["sample_data_token"],
            "pointcloud_path": str(path.relative_to(output_dir.parent.parent)),
            "coordinate_frame": "ego",
            "num_sweeps": num_sweeps,
            "num_points": int(count),
            "num_ground_points": int(ground_mask.sum()),
        }

    # ── 3. マスク処理と Box Fitting ──────────────────────────────────────

    def process_frame_instances(
        self,
        instances: list[dict[str, Any]],
        depth_result: dict[str, Any],
        output_dir: Path,
        *,
        frame: dict[str, Any] | None = None,
        mask_params: dict[str, Any] | None = None,
        depth_params: dict[str, Any] | None = None,
        use_lidar: bool = False,
        stored_points_max: int = 500,
        stub_delay_sec: float | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """1 フレーム分のインスタンスを処理する（本実装と同じ入口）.

        スタブではマスクの大きさから距離を作り、それらしい点群を散らす。
        マスクが小さいものは「点が少なすぎる」として失敗させ、
        UI が失敗ケースを表示できるようにしておく。
        """
        time.sleep(DEFAULT_STUB_DELAY_SEC if stub_delay_sec is None else stub_delay_sec)

        results: list[dict[str, Any]] = []
        for instance in instances:
            mask_rle = instance["mask_rle"]
            xmin, ymin, xmax, ymax = rle_bbox(mask_rle)
            area = max(0, (xmax - xmin) * (ymax - ymin))

            base = {
                "instance_tracking_2d_id": instance["instance_tracking_2d_id"],
                "sample_data_token": instance["sample_data_token"],
                "track_id": str(instance["track_id"]),
                "label": instance["label"],
                "mask_rle_closed": mask_rle,
            }
            if area == 0:
                results.append({**base, "status": "no_points",
                                "num_points_depth": 0, "num_points_lidar": 0})
                continue

            rng = np.random.default_rng(int(hashlib.md5(
                f"{instance['sample_data_token']}:{instance['track_id']}".encode()
            ).hexdigest()[:8], 16))
            distance = float(np.clip(20000.0 / max(area, 1), 3.0, 60.0))
            num_depth = int(np.clip(area // 20, 0, 8000))
            num_lidar = int(np.clip(area // 400, 0, 600)) if use_lidar else 0

            if num_depth < 10:
                results.append({**base, "status": "too_few_points",
                                "num_points_depth": num_depth,
                                "num_points_lidar": num_lidar})
                continue

            center = np.array([distance, rng.uniform(-6, 6), rng.uniform(0.3, 1.5)])
            size = np.array([1.9, 4.6, 1.7])
            depth_points = center + rng.normal(0, size / 4.0, size=(num_depth, 3))
            lidar_points = (
                center + rng.normal(0, size / 5.0, size=(num_lidar, 3))
                if num_lidar else np.empty((0, 3))
            )
            results.append({
                **base,
                "status": "fitted",
                "points_depth_ego": points_to_json(
                    downsample_to_max(depth_points, stored_points_max)
                ),
                "points_lidar_ego": points_to_json(
                    downsample_to_max(lidar_points, stored_points_max)
                ) if num_lidar else None,
                "num_points_depth": num_depth,
                "num_points_lidar": num_lidar,
                "center_ego": [round(float(v), 3) for v in center],
                "size_wlh": [float(v) for v in size],
                "yaw_ego": float(rng.uniform(-np.pi, np.pi)),
                "fitting_score": round(float(rng.uniform(0.4, 0.95)), 3),
            })
        return results
