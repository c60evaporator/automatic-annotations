"""保存済みの深度マップから、インスタンス点群を再生成して再フィルタする.

パラメータ調整のたびに数分のパイプラインを回し直さずに済むようにする。
点群は「深度マップをクロージング後マスクで切り出したもの」なので、
保存済みの .npz とマスクがあれば完全に再現できる。追加保存は不要。

**モデルはロードしない。** ROR / DBSCAN に DA3 は要らないので、
model_registry を経由すると調整のたびに VRAM を占有することになる。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from common.transform3d import ego_to_ego
from app.services.depth_ops import (
    depth_map_to_point_cloud,
    filter_outliers,
    scaled_nb_points,
    instance_points_from_depth,
    resize_mask_nearest,
    rle_to_depth_mask,
)
from common.point_ops import downsample_pair, downsample_to_max, points_to_json
from common.transform3d import camera_to_ego, scale_intrinsic

logger = get_logger(__name__)


def _reduce_pair(
    raw: np.ndarray, filtered: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """フィルタ前後の点群を表示・保存用に間引く.

    - **フィルタ後は推論時と同じ ``downsample_to_max``** で間引く。
      推論時の保存結果と Apply 後の表示を一致させるため。
      ``downsample_pair`` のように raw 基準のボクセルを使うと、背景を
      巻き込んで広がった raw に引きずられ、本来 500 点以下で間引き
      不要な kept まで潰れる（144 点 → 7 点の例あり）。
    - フィルタ前は従来どおり ``downsample_pair`` のボクセルで間引く
      （raw 同士の見え方は変えない）。

    NOTE: 両者のボクセルサイズが揃わなくなるため、Raw 表示と
      ROR/DBSCAN 表示を切り替えると、まれにフィルタ後のほうが
      点が多く見える場合がある（実際の点数は raw ≥ kept）。
    """
    reduced_raw, _ = downsample_pair(raw, filtered, max_points)
    filtered = np.asarray(filtered, dtype=np.float64).reshape(-1, 3)
    return reduced_raw, downsample_to_max(filtered, max_points)


def _load_depth(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """深度マップ .npz を読む（内部パラメータと空マスクは無い場合もある）."""
    with np.load(path) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
        intrinsic = (
            np.asarray(data["intrinsic"], dtype=np.float64)
            if "intrinsic" in data.files else None
        )
        non_sky = (
            np.asarray(data["non_sky"], dtype=bool)
            if "non_sky" in data.files else None
        )
    return depth, intrinsic, non_sky


def refilter_frame(
    depth_path: Path,
    instances: list[dict[str, Any]],
    *,
    calibrated_sensor: dict[str, Any],
    depth_params: dict[str, Any],
    nb_points_ratio: dict[str, float] | None = None,
    stored_points_max: int = 500,
    max_depth: float | None = None,
    camera_intrinsic: Any | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    lidar_path: Path | None = None,
    lidar_params: dict[str, Any] | None = None,
    min_lidar_points: int = 0,
    camera_ego_pose: dict[str, Any] | None = None,
    reference_ego_pose: dict[str, Any] | None = None,
    depth_correction_method: str | None = None,
    max_lidar_points_for_depth: int = 0,
    lshape_params: dict[str, Any] | None = None,
    lidar_origin_xy: tuple[float, float] | None = None,
    distance_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """1 フレーム分のインスタンスを、新しいパラメータで作り直す.

    Args:
        camera_intrinsic / image_width / image_height:
            .npz に内部パラメータが入っていない場合のフォールバック。
            元画像の K を深度マップの解像度へスケールして使う

    Args:
        lidar_path: そのサンプルの LiDAR .npz。渡すと LiDAR 側も作り直す
        min_lidar_points: この数に届かない LiDAR 点群は捨てる
        depth_correction_method: LiDAR を基準に深度点群を補正する方式。
            **外れ値除去の条件を変えると LiDAR 点群が変わり、補正係数も変わる**
            ので、再フィルタでも作り直す
        max_lidar_points_for_depth: LiDAR 点がこの数以上なら深度点群を使わない
        lshape_params: L 字フィルタの共通設定（run の box_fitting_params["lshape"]）
        lidar_origin_xy: LiDAR の位置（基準 ego 座標の XY）
        distance_params: 距離フィルタの設定（box_fitting_params["distance_filter"]
            に想定サイズ ``box_sizes`` を足したもの）

    Returns:
        インスタンスごとの ``{id, num_points_raw, num_points_kept,
        points_raw_ego, points_filtered_ego,
        num_lidar_raw, num_lidar_kept,
        points_lidar_raw_ego, points_lidar_filtered_ego,
        depth_used, depth_correction,
        points_corrected_raw_ego, points_corrected_filtered_ego,
        points_lshape_ego, points_corrected_lshape_ego,
        points_lidar_lshape_ego, lshape_filter,
        points_distance_ego, points_corrected_distance_ego,
        points_lidar_distance_ego, distance_filter}``

    **推論（BoxFittingPipeline.process_frame_instances）と同じ順序で処理する。**
    ROR/DBSCAN → LiDAR の L 字フィルタ → LiDAR の距離フィルタ →
    深度を使うかの判定 → 補正 → 深度の L 字フィルタ → 深度の距離フィルタ。
    順序が違うと、推論時と違う点群になる

    NOTE: LiDAR 側は **深度マップとは独立**に作り直す。
    点の出所が違うので、外れ値除去のパラメータも別に渡すこと。
    """
    depth, intrinsic, non_sky = _load_depth(depth_path)
    depth_height, depth_width = depth.shape

    if intrinsic is None:
        # 保存時の内部パラメータが無い run（古い形式）向けの経路。
        # 元画像の K を深度側の解像度へ合わせる
        if camera_intrinsic is None or not image_width or not image_height:
            raise ValueError(
                "深度マップに内部パラメータが無く、"
                "フォールバック用の camera_intrinsic も渡されていません"
            )
        intrinsic = scale_intrinsic(
            camera_intrinsic, depth_width / image_width, depth_height / image_height
        )

    if non_sky is not None and non_sky.shape != depth.shape:
        non_sky = resize_mask_nearest(non_sky, depth_height, depth_width)

    # 全画素の点群は 1 回だけ作り、各インスタンスのマスクで切り出す
    all_points = depth_map_to_point_cloud(depth, intrinsic, depth_threshold=max_depth)

    # LiDAR は 1 フレームにつき 1 回だけ読む（地面は除く）
    lidar_points = np.empty((0, 3))
    if lidar_path is not None and Path(lidar_path).exists():
        with np.load(lidar_path) as data:
            points = np.asarray(data["points"], dtype=np.float64)
            if "ground_mask" in data.files:
                points = points[~np.asarray(data["ground_mask"], dtype=bool)]
            lidar_points = points[:, :3]

    # 小物体用の倍率。どのインスタンスに適用するかは**推論時の判定**
    # （box_fittings.is_small_mask）に従う。クロージング後のマスクからは
    # 元の短辺が分からないため、ここで判定し直さない
    small_nb_ratio = float(depth_params.get("small_mask_nb_points_ratio", 1.0))

    # L 字フィルタ（推論と同じ解釈をするため、設定の組み立ては共通の関数で行う）
    from app.services.lshape_filter import (
        lshape_applies, lshape_filter, lshape_settings, sensor_origin_xy,
    )
    depth_lshape = lshape_settings(
        lshape_params, depth_params, margin_ratio_default=0.0
    )
    lidar_lshape = lshape_settings(lshape_params, lidar_params)
    camera_origin = sensor_origin_xy(
        calibrated_sensor, camera_ego_pose, reference_ego_pose
    )
    from app.services.distance_filter import distance_filter, distance_settings
    distance_general = distance_params or {}
    box_sizes = distance_general.get("box_sizes")

    results: list[dict[str, Any]] = []
    for instance in instances:
        small = bool(instance.get("is_small_mask"))
        mask = rle_to_depth_mask(
            instance["mask_rle_closed"], depth_height, depth_width
        )
        filtered, raw = instance_points_from_depth(
            all_points, mask,
            common_mask=non_sky,
            ror_nb_points=int(depth_params.get("ror_nb_points", 0)),
            ror_radius=float(depth_params.get("ror_radius", 0.0)),
            dbscan_eps=float(depth_params.get("dbscan_eps", 0.0)),
            dbscan_min_samples=int(depth_params.get("dbscan_min_samples", 0)),
            label=instance.get("label"),
            nb_points_ratio=nb_points_ratio,
            small_mask_ratio=small_nb_ratio if small else 1.0,
        )

        def to_reference(points_camera: np.ndarray) -> np.ndarray:
            """カメラ座標 → 基準 ego 座標.

            パイプラインが保存する点群と同じ座標系に揃える。
            揃えないとカメラ間で時刻差ぶんずれる（自車 10 m/s・25 ms で 0.25 m）
            """
            points = camera_to_ego(
                points_camera,
                calibrated_sensor["translation"], calibrated_sensor["rotation"],
            )
            if reference_ego_pose:
                points = ego_to_ego(
                    points, camera_ego_pose or {}, reference_ego_pose
                )
            return points

        raw_ego = to_reference(raw)
        filtered_ego = to_reference(filtered)
        # 前後で同じボクセルサイズを使う。別々に間引くと、広がりの大きい
        # フィルタ前が粗くなり「フィルタして点が増えた」ように見える
        reduced_raw, reduced_filtered = _reduce_pair(
            raw_ego, filtered_ego, stored_points_max
        )

        # --- LiDAR 側 -----------------------------------------------
        lidar_raw = np.empty((0, 3))
        lidar_kept = np.empty((0, 3))
        if lidar_points.shape[0] and calibrated_sensor.get("camera_intrinsic"):
            from app.services.lidar_ops import instance_lidar_points

            lidar_raw = instance_lidar_points(
                lidar_points,
                resize_mask_nearest(mask, int(image_height), int(image_width))
                if image_height and image_width else mask,
                camera_calib=calibrated_sensor,
                intrinsic=np.asarray(
                    calibrated_sensor["camera_intrinsic"], dtype=np.float64
                ),
                image_width=int(image_width or 0),
                image_height=int(image_height or 0),
                camera_ego_pose=camera_ego_pose,
                reference_ego_pose=reference_ego_pose,
            )
            if lidar_raw.shape[0]:
                lidar_kept = filter_outliers(
                    lidar_raw,
                    ror_nb_points=scaled_nb_points(
                        int((lidar_params or {}).get("ror_nb_points", 0)),
                        instance.get("label"), nb_points_ratio,
                    ),
                    ror_radius=float((lidar_params or {}).get("ror_radius", 0.0)),
                    dbscan_eps=float((lidar_params or {}).get("dbscan_eps", 0.0)),
                    dbscan_min_samples=int(
                        (lidar_params or {}).get("dbscan_min_samples", 0)
                    ),
                )
                if lidar_kept.shape[0] < min_lidar_points:
                    lidar_kept = np.empty((0, 3))

        lidar_raw_reduced, lidar_kept_reduced = _reduce_pair(
            lidar_raw, lidar_kept, stored_points_max
        )

        # --- L 字フィルタ（LiDAR） --------------------------------------
        # 以降（深度を使うかの判定・補正の基準）はフィルタ後の点群で行う
        lshape_info: dict[str, Any] = {}
        applies = lshape_applies(lshape_params, instance.get("label"))
        lidar_used = lidar_kept
        if (applies and lidar_lshape and lidar_origin_xy is not None
                and lidar_kept.shape[0]):
            keep, lshape_info["lidar"] = lshape_filter(
                lidar_kept, lidar_origin_xy, **lidar_lshape
            )
            lidar_used = lidar_kept[keep]
        # 表示用に L 字フィルタ直後の LiDAR 点群を取っておく。
        # NOTE: lidar_lshape（L 字フィルタの設定）と名前を分けること。
        # 上書きすると、次のインスタンスで設定の代わりに配列を見て落ちる
        lidar_after_lshape = lidar_used

        # --- 距離フィルタ（LiDAR） --------------------------------------
        distance_info: dict[str, Any] = {}
        distance = distance_settings(
            distance_general, box_sizes, instance.get("label")
        )
        if distance and lidar_origin_xy is not None and lidar_used.shape[0]:
            keep, distance_info["lidar"] = distance_filter(
                lidar_used, lidar_origin_xy, **distance
            )
            lidar_used = lidar_used[keep]

        # --- 深度点群の補正（1 段目） ----------------------------------
        # 外れ値除去の条件を変えると LiDAR 点群が変わるので、
        # 補正係数も作り直す（推論時の値を流用しない）。
        # 推論と同じく 2 段で行う。1 段目（クロージング後マスク全体の画素）は
        # フィルタの判定にだけ使い、最終の補正は 2 段目（残った点の画素）
        initial_correction = None
        lidar_camera: np.ndarray | None = None
        initial_corrected_ego: np.ndarray | None = None
        # 推論と同じ条件（深度点が 1 つも残らなければ深度は使わない）
        depth_used = filtered.shape[0] > 0 and not (
            max_lidar_points_for_depth > 0
            and lidar_used.shape[0] >= max_lidar_points_for_depth
        )
        if depth_used and depth_correction_method and lidar_used.shape[0]:
            from app.services.depth_correction import (
                apply_correction, fit_correction, sample_depth_at,
            )
            from app.services.lidar_ops import ego_to_camera

            lidar_camera = ego_to_camera(
                ego_to_ego(
                    lidar_used, reference_ego_pose or {}, camera_ego_pose or {}
                ) if reference_ego_pose else lidar_used,
                calibrated_sensor,
            )
            z_lidar, z_pred = sample_depth_at(
                lidar_camera, depth, intrinsic, mask=mask
            )
            initial_correction = fit_correction(
                z_lidar, z_pred, depth_correction_method
            )
            if initial_correction is not None:
                initial_corrected_ego = to_reference(
                    apply_correction(filtered, initial_correction)
                )

        # 判定に使う点群（1 段目の補正ができれば補正後）
        judged = (
            initial_corrected_ego if initial_corrected_ego is not None
            else filtered_ego
        )
        # ROR/DBSCAN 後の点に対する、各フィルタで残すかのマスク
        keep_lshape: np.ndarray | None = None
        keep_final = np.ones(filtered_ego.shape[0], dtype=bool)

        # --- L 字フィルタ（深度） ---------------------------------------
        if applies and depth_lshape and filtered_ego.shape[0]:
            keep, lshape_info["depth"] = lshape_filter(
                judged, camera_origin, **depth_lshape,
            )
            keep_lshape = keep
            keep_final &= keep

        # --- 距離フィルタ（深度） ---------------------------------------
        # L 字フィルタ後（適用しなければ ROR/DBSCAN 後）に、同じ規則で当てる
        keep_distance_applied = False
        if distance and filtered_ego.shape[0]:
            index = np.flatnonzero(keep_final)
            keep, distance_info["depth"] = distance_filter(
                judged[index], camera_origin, **distance,
            )
            keep_final[index[~keep]] = False
            keep_distance_applied = True

        # --- 深度点群の補正（2 段目） ----------------------------------
        correction = None
        corrected_raw_reduced = corrected_filtered_reduced = None
        corrected_lshape = corrected_distance = None
        if lidar_camera is not None:
            from app.services.depth_correction import (
                apply_correction, refine_correction,
            )
            correction = refine_correction(
                lidar_camera, depth, intrinsic, filtered[keep_final],
                depth_correction_method, initial_correction,
            )
            if correction is not None:
                corrected_filtered_ego = to_reference(
                    apply_correction(filtered, correction)
                )
                corrected_raw_reduced, corrected_filtered_reduced = _reduce_pair(
                    to_reference(apply_correction(raw, correction)),
                    corrected_filtered_ego,
                    stored_points_max,
                )
                if keep_lshape is not None:
                    corrected_lshape = corrected_filtered_ego[keep_lshape]
                if keep_distance_applied:
                    corrected_distance = corrected_filtered_ego[keep_final]

        filtered_lshape = (
            filtered_ego[keep_lshape] if keep_lshape is not None else None
        )
        filtered_distance = (
            filtered_ego[keep_final] if keep_distance_applied else None
        )

        def _stored(points: np.ndarray | None) -> dict[str, Any] | None:
            # 推論時の保存と同じ間引き方にする（表示が一致するように）
            if points is None or points.shape[0] == 0:
                return None
            return points_to_json(downsample_to_max(points, stored_points_max))

        results.append({
            # L 字フィルタ後。適用しなかった場合は None（ROR/DBSCAN 後と同じ）
            "points_lshape_ego": _stored(filtered_lshape),
            "points_corrected_lshape_ego": _stored(corrected_lshape),
            "points_lidar_lshape_ego": (
                _stored(lidar_after_lshape) if "lidar" in lshape_info else None
            ),
            "lshape_filter": lshape_info or None,
            # 距離フィルタ後。適用しなかった場合は None（L 字フィルタ後と同じ）
            "points_distance_ego": _stored(filtered_distance),
            "points_corrected_distance_ego": _stored(corrected_distance),
            "points_lidar_distance_ego": (
                _stored(lidar_used) if "lidar" in distance_info else None
            ),
            "distance_filter": distance_info or None,
            "depth_used": depth_used,
            "depth_correction": correction,
            "points_corrected_raw_ego": (
                points_to_json(corrected_raw_reduced)
                if corrected_raw_reduced is not None else None
            ),
            "points_corrected_filtered_ego": (
                points_to_json(corrected_filtered_reduced)
                if corrected_filtered_reduced is not None else None
            ),
            "num_lidar_raw": int(lidar_raw.shape[0]),
            "num_lidar_kept": int(lidar_kept.shape[0]),
            "points_lidar_raw_ego": (
                points_to_json(lidar_raw_reduced) if lidar_raw.shape[0] else None
            ),
            "points_lidar_filtered_ego": (
                points_to_json(lidar_kept_reduced) if lidar_kept.shape[0] else None
            ),
            "id": instance["id"],
            "track_id": instance.get("track_id"),
            "label": instance.get("label"),
            "is_small_mask": small,
            "num_points_raw": int(raw.shape[0]),
            "num_points_kept": int(filtered.shape[0]),
            "points_raw_ego": points_to_json(reduced_raw),
            "points_filtered_ego": points_to_json(reduced_filtered),
        })
    return results


def _lidar_origin(frame: dict[str, Any]) -> tuple[float, float] | None:
    """フレームに付いた LiDAR の位置情報から、基準 ego 座標の XY を求める."""
    from app.services.lshape_filter import sensor_origin_xy

    calib = frame.get("lidar_calibrated_sensor") or {}
    if not calib.get("translation"):
        return None
    return sensor_origin_xy(
        calib, frame.get("lidar_ego_pose"), frame.get("reference_ego_pose")
    )


def refilter(
    derived_root: Path,
    frames: list[dict[str, Any]],
    *,
    depth_params: dict[str, Any],
    nb_points_ratio: dict[str, float] | None = None,
    lidar_params: dict[str, Any] | None = None,
    min_lidar_points: int = 0,
    depth_correction_method: str | None = None,
    max_lidar_points_for_depth: int = 0,
    stored_points_max: int = 500,
    max_depth: float | None = None,
    lshape_params: dict[str, Any] | None = None,
    distance_params: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """複数フレームをまとめて再フィルタする.

    Returns:
        (インスタンスごとの結果, 所要秒)
    """
    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    for frame in frames:
        path = derived_root / frame["depth_path"]
        if not path.exists():
            logger.warning("depth map not found: %s", path)
            continue
        try:
            results.extend(refilter_frame(
                path, frame["instances"],
                calibrated_sensor=frame["calibrated_sensor"],
                depth_params=depth_params,
                nb_points_ratio=nb_points_ratio,
                lidar_params=lidar_params,
                min_lidar_points=min_lidar_points,
                depth_correction_method=depth_correction_method,
                max_lidar_points_for_depth=max_lidar_points_for_depth,
                # LiDAR は sample 単位。フレームごとに渡ってくる
                lidar_path=(
                    derived_root / frame["lidar_path"]
                    if frame.get("lidar_path") else None
                ),
                camera_ego_pose=frame.get("ego_pose"),
                reference_ego_pose=frame.get("reference_ego_pose"),
                lshape_params=lshape_params,
                distance_params=distance_params,
                lidar_origin_xy=_lidar_origin(frame),
                stored_points_max=stored_points_max,
                max_depth=max_depth,
                camera_intrinsic=(frame.get("calibrated_sensor") or {}).get(
                    "camera_intrinsic"
                ),
                image_width=frame.get("width"),
                image_height=frame.get("height"),
            ))
        except Exception as exc:  # noqa: BLE001
            # 1 フレームの失敗で全体を落とさない
            logger.warning("refilter failed for %s: %s", frame.get("depth_path"), exc)

    return results, time.perf_counter() - started
