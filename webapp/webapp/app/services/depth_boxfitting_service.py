"""深度推定＋Box Fitting run の保存と読み出し（webapp 側の書き込み主体）.

Detection2D / Instance Tracking と同じ形に揃えてある:
  - ジョブ完了時に 1 トランザクションでまとめて保存
  - 追加（履歴保持）＋ 上限プルーニング
  - キャンセル・失敗も status 付きで残す

このステップ固有の事情が 2 つある:
  1. 深度マップと LiDAR 点群を DERIVED_ROOT にファイルとして持つため、
     run 削除時にディレクトリごと消す必要がある（DB の CASCADE では消えない）
  2. 成功したフィッティングを SampleAnnotation（global 座標）へ起こす
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import read_only_session, session_scope
from app.models.ann_intermediate import (
    RUN_STATUS_CANCELLED,
    RUN_STATUS_FAILED,
    RUN_STATUS_SUCCEEDED,
)
from app.models.annotation import SOURCE_AUTO, Category, Instance, SampleAnnotation
from app.models.scene import Sample
from app.models.sensor import SampleData
from app.repositories.depth_boxfitting import DepthBoxFittingRepository
from app.repositories.instance_tracking import InstanceTrackingRepository
from app.repositories.sensor import SensorRepository
from app.services.geometry.pointcloud import transform_ego_to_global
from app.services.geometry.transform import (
    normalize_quaternion,
    quaternion_multiply,
    yaw_to_quaternion,
)
from app.services.label_service import label_to_nusc_category
from app.services.sweep_service import select_lidar_sweeps

logger = get_logger(__name__)

_JOB_STATUS_MAP = {
    "succeeded": RUN_STATUS_SUCCEEDED,
    "failed": RUN_STATUS_FAILED,
    "cancelled": RUN_STATUS_CANCELLED,
}


def run_derived_dir(params_id: str) -> Path:
    """run の派生ファイル置き場.

    深度マップと LiDAR 点群を run ごとのディレクトリにまとめる。
    こうしておくと、削除が 1 回の rmtree で済む。
    """
    return get_settings().DERIVED_ROOT / "boxfitting" / params_id


# ── リクエストの組み立て ──────────────────────────────────────────────────────

def build_boxfitting_payload(
    dataset_id: str,
    scene_token: str,
    dataroot: str,
    *,
    params_id: str,
    tracking_run_id: str,
    use_lidar: bool,
    num_lidar_sweeps: int,
    mask_params: dict[str, Any],
    depth_params: dict[str, Any],
    lidar_params: dict[str, Any],
    box_fitting_params: dict[str, Any] | None = None,
    stub_delay_sec: float | None = None,
) -> dict[str, Any]:
    """推論サーバーへ送るリクエストを組み立てる.

    深度推定は全 sample × 全カメラのキーフレームが対象。
    LiDAR は sample ごとに 1 件（LIDAR_TOP のキーフレーム）。
    """
    settings = get_settings()

    with read_only_session() as session:
        tracking_repo = InstanceTrackingRepository(session)
        tracking_run = tracking_repo.get_run(tracking_run_id)
        if tracking_run is None:
            raise ValueError(
                f"Instance Tracking run が見つかりません: {tracking_run_id}"
            )
        instances_by_frame = tracking_repo.list_instances_by_run(
            tracking_run_id, include_mask=True
        )
        sensor_repo = SensorRepository(session)
        frames = sensor_repo.list_frames_by_scene(scene_token, keyframe_only=True)
        # LiDAR は sweep も要る（キーフレームだけでは統合できない）
        all_frames = sensor_repo.list_frames_by_scene(scene_token, keyframe_only=False)

    camera_frames = [f for f in frames if f["modality"] == "camera"]
    lidar_frames = [f for f in frames if f["modality"] == "lidar"]
    # sample ごとに直近 num_lidar_sweeps 件（末尾がキーフレーム）
    sweeps_by_sample = select_lidar_sweeps(all_frames, num_lidar_sweeps)

    def frame_payload(frame: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_data_token": frame["token"],
            "sample_token": frame["sample_token"],
            "filename": frame["filename"],
            "channel": frame["channel"],
            "sample_idx": frame.get("sample_idx", 0),
            "timestamp": frame["timestamp"],
            "is_key_frame": frame.get("is_key_frame", True),
            "width": frame["width"],
            "height": frame["height"],
            "ego_pose": frame["ego_pose"],
            "calibrated_sensor": frame["calibrated_sensor"],
        }

    # トラッキングのインスタンス（マスクの供給元）。
    # 区間境界には prompt / propagated の 2 件があるので prompt を優先する
    from app.streamlit.components.instance_tracking_viewer import preferred_instances

    instances_payload = [
        {
            "instance_tracking_2d_id": inst["id"],
            "sample_data_token": token,
            "track_id": inst["track_id"],
            "label": inst["label"],
            "mask_rle": inst["mask_rle"],
        }
        for token, insts in instances_by_frame.items()
        for inst in preferred_instances(insts)
    ]

    return {
        "dataroot": dataroot,
        "camera_frames": [frame_payload(f) for f in camera_frames],
        "lidar_frames": [frame_payload(f) for f in lidar_frames],
        "lidar_sweeps": [
            frame_payload(f)
            for group in sweeps_by_sample.values() for f in group
        ],
        "instances": instances_payload,
        "use_lidar": use_lidar,
        "num_lidar_sweeps": num_lidar_sweeps,
        "mask_params": mask_params,
        "depth_params": depth_params,
        "lidar_params": lidar_params,
        "box_fitting_params": box_fitting_params or {},
        # 保存する点群の上限（間引き後）。推論サーバー側で間引いて返す
        "stored_points_max": settings.BOXFIT_STORED_POINTS_MAX,
        # 派生ファイルの出力先（推論サーバーも /derived を共有マウントしている）。
        # run を先に作って id を確定させてから投げるので、
        # 推論サーバーは run 専用のディレクトリへ直接書ける
        "params_id": params_id,
        "output_dir": f"boxfitting/{params_id}",
        "depth_downscale": settings.DEPTH_MAP_DOWNSCALE,
        "stub_delay_sec": stub_delay_sec,
    }


# ── 保存 ──────────────────────────────────────────────────────────────────────

def create_pending_run(
    dataset_id: str,
    scene_token: str,
    *,
    tracking_run_id: str,
    use_lidar: bool,
    num_lidar_sweeps: int,
    mask_params: dict[str, Any],
    depth_params: dict[str, Any],
    lidar_params: dict[str, Any],
    box_fitting_params: dict[str, Any] | None = None,
    model_name: str = "",
) -> str:
    """推論を投げる前に run を作成して id を返す（status='running'）.

    深度マップと LiDAR 点群は推論サーバーが DERIVED_ROOT へ直接書くため、
    書き込み先ディレクトリ名になる params_id を先に確定させる必要がある。
    完了後に finalize_run() で結果と最終状態を書き込む。
    """
    settings = get_settings()
    with session_scope() as session:
        return DepthBoxFittingRepository(session).create_run(
            dataset_id, scene_token,
            instance_tracking_2d_params_id=tracking_run_id,
            model_name=model_name or settings.DEPTH_MODEL_NAME,
            use_lidar=use_lidar,
            num_lidar_sweeps=num_lidar_sweeps,
            mask_params=mask_params,
            depth_params=depth_params,
            lidar_params=lidar_params,
            box_fitting_params=box_fitting_params,
        )


def finalize_run(
    dataset_id: str,
    scene_token: str,
    params_id: str,
    *,
    job: dict[str, Any],
    depth_estimations: list[dict[str, Any]],
    lidar_pointclouds: list[dict[str, Any]],
    box_fittings: list[dict[str, Any]],
) -> int:
    """完了したジョブの結果を、作成済みの run へ書き込む.

    Returns:
        生成した SampleAnnotation の件数。
    """
    status = _JOB_STATUS_MAP.get(job.get("status", ""), RUN_STATUS_FAILED)

    with session_scope() as session:
        repo = DepthBoxFittingRepository(session)
        n_depth = repo.save_depth_estimations(params_id, dataset_id, depth_estimations)
        n_lidar = repo.save_lidar_pointclouds(params_id, dataset_id, lidar_pointclouds)
        n_box = repo.save_box_fittings(params_id, dataset_id, box_fittings)
        repo.finish_run(
            params_id,
            status=status,
            num_inferences=int(job.get("processed", 0)),
            num_boxes=sum(
                1 for b in box_fittings if b.get("center_ego") is not None
            ),
            inference_time=(job.get("result") or {}).get("inference_time"),
        )

    logger.info(
        "finalized boxfitting run %s: depth=%d lidar=%d box=%d status=%s",
        params_id, n_depth, n_lidar, n_box, status,
    )

    created = 0
    if status == RUN_STATUS_SUCCEEDED:
        created = materialize_annotations(dataset_id, params_id)
        logger.info("materialized %d annotations for run %s", created, params_id)

    pruned = prune_runs(dataset_id, scene_token)
    if pruned:
        logger.info("pruned %d old runs", len(pruned))
    return created


# ── SampleAnnotation の生成 ───────────────────────────────────────────────────

def materialize_annotations(dataset_id: str, params_id: str) -> int:
    """成功したフィッティングを SampleAnnotation（global 座標）へ起こす.

    Box Fitting の結果は ego 座標なので、フレームの ego_pose を使って
    global へ変換する。track_id はカメラを跨がないため、
    同じ物体が複数カメラに写ると別々の Instance になる（重複を許容する方針）。

    Returns:
        作成した SampleAnnotation の件数。
    """
    with session_scope() as session:
        repo = DepthBoxFittingRepository(session)
        fittings = repo.list_unlinked_fittings(params_id)
        if not fittings:
            return 0

        # 必要な参照をまとめて引く
        tokens = {f["sample_data_token"] for f in fittings}
        frame_rows = session.execute(
            _frame_lookup_stmt(list(tokens))
        ).mappings().all()
        frames = {r["token"]: dict(r) for r in frame_rows}

        categories = {
            name: token for token, name in session.execute(
                _category_lookup_stmt(dataset_id)
            )
        }

        # (channel, track_id) ごとに Instance を 1 つ作る。
        # track_id はカメラ内でのみ一意なので、channel を鍵に含めないと
        # 別カメラの別物体が同じ Instance にまとまってしまう
        instance_tokens: dict[tuple[str, str], str] = {}
        # (box_fitting_id, annotation_token) を溜めておき、flush 後にまとめて更新する。
        # sessionmaker は autoflush=False なので、add した直後に Core の UPDATE を
        # 流すと参照先の行がまだ INSERT されておらず FK 違反になる
        links: list[tuple[str, str]] = []
        created = 0

        for fit in sorted(
            fittings, key=lambda f: frames.get(f["sample_data_token"], {}).get("timestamp", 0)
        ):
            frame = frames.get(fit["sample_data_token"])
            if frame is None:
                continue

            category_name = label_to_nusc_category(fit["label"])
            category_token = categories.get(category_name or "")
            if category_token is None:
                # データセットに無いカテゴリは 3D ボックスにできない
                logger.warning(
                    "category not found for label %r (-> %r); skipped",
                    fit["label"], category_name,
                )
                continue

            key = (frame["channel"], str(fit["track_id"]))
            if key not in instance_tokens:
                token = str(uuid.uuid4())
                session.add(Instance(
                    token=token, dataset_id=dataset_id,
                    category_token=category_token, nbr_annotations=0,
                    source=SOURCE_AUTO,
                ))
                instance_tokens[key] = token

            center_global = transform_ego_to_global(
                np.asarray(fit["center_ego"], dtype=np.float64).reshape(1, 3),
                frame["ego_translation"], frame["ego_rotation"],
            )[0]
            # ego 座標の yaw を global の回転へ合成する
            rotation_global = normalize_quaternion(quaternion_multiply(
                normalize_quaternion(frame["ego_rotation"]),
                yaw_to_quaternion(fit["yaw_ego"] or 0.0),
            ))

            annotation_token = str(uuid.uuid4())
            session.add(SampleAnnotation(
                token=annotation_token, dataset_id=dataset_id,
                sample_token=frame["sample_token"],
                instance_token=instance_tokens[key],
                translation=center_global.tolist(),
                rotation=rotation_global.tolist(),
                size=fit["size_wlh"],
                num_lidar_pts=int(fit.get("num_points_lidar") or 0),
                num_radar_pts=0,
                source=SOURCE_AUTO,
                depth_estimation_params_id=params_id,
                score=fit.get("fitting_score"),
            ))
            links.append((fit["id"], annotation_token))
            created += 1

        # ここで Instance と SampleAnnotation が INSERT される
        session.flush()
        for box_fitting_id, annotation_token in links:
            repo.link_annotation(box_fitting_id, annotation_token)
        _update_instance_chains(session, list(instance_tokens.values()))

    return created


def _frame_lookup_stmt(tokens: list[str]):
    from sqlalchemy import select

    from app.models.sensor import CalibratedSensor, EgoPose, Sensor

    return (
        select(
            SampleData.token,
            SampleData.sample_token,
            SampleData.timestamp,
            Sensor.channel,
            EgoPose.translation.label("ego_translation"),
            EgoPose.rotation.label("ego_rotation"),
        )
        .join(CalibratedSensor,
              CalibratedSensor.token == SampleData.calibrated_sensor_token)
        .join(Sensor, Sensor.token == CalibratedSensor.sensor_token)
        .join(EgoPose, EgoPose.token == SampleData.ego_pose_token)
        .where(SampleData.token.in_(tokens))
    )


def _category_lookup_stmt(dataset_id: str):
    from sqlalchemy import select

    return select(Category.token, Category.name).where(
        Category.dataset_id == dataset_id
    )


def _update_instance_chains(session, instance_tokens: list[str]) -> None:
    """Instance の prev/next チェーンと件数を埋める.

    nuScenes 形式でエクスポートするとき、first/last/nbr が揃っていないと
    devkit 側の想定から外れる。
    """
    from sqlalchemy import select, update

    for instance_token in instance_tokens:
        rows = session.execute(
            select(SampleAnnotation.token)
            .join(Sample, Sample.token == SampleAnnotation.sample_token)
            .where(SampleAnnotation.instance_token == instance_token)
            .order_by(Sample.timestamp)
        ).scalars().all()
        if not rows:
            continue
        for index, token in enumerate(rows):
            session.execute(
                update(SampleAnnotation.__table__)
                .where(SampleAnnotation.__table__.c.token == token)
                .values(
                    prev=rows[index - 1] if index > 0 else None,
                    next=rows[index + 1] if index + 1 < len(rows) else None,
                )
            )
        session.execute(
            update(Instance.__table__)
            .where(Instance.__table__.c.token == instance_token)
            .values(
                nbr_annotations=len(rows),
                first_annotation_token=rows[0],
                last_annotation_token=rows[-1],
            )
        )


# ── 読み出し ──────────────────────────────────────────────────────────────────

def resolve_display_run(dataset_id: str, scene_token: str) -> tuple[str | None, str]:
    with read_only_session() as session:
        return DepthBoxFittingRepository(session).resolve_display_run(
            dataset_id, scene_token
        )


def get_run(params_id: str) -> dict[str, Any] | None:
    with read_only_session() as session:
        return DepthBoxFittingRepository(session).get_run(params_id)


def list_runs(dataset_id: str, scene_token: str) -> list[dict[str, Any]]:
    with read_only_session() as session:
        return DepthBoxFittingRepository(session).list_runs(dataset_id, scene_token)


def list_input_tracking_runs(
    dataset_id: str, scene_token: str
) -> list[dict[str, Any]]:
    with read_only_session() as session:
        return DepthBoxFittingRepository(session).list_tracking_runs_for_input(
            dataset_id, scene_token
        )


def load_depth_estimations(params_id: str) -> dict[str, dict[str, Any]]:
    with read_only_session() as session:
        return DepthBoxFittingRepository(session).list_depth_estimations_by_run(
            params_id
        )


def load_lidar_pointclouds(params_id: str) -> dict[str, dict[str, Any]]:
    with read_only_session() as session:
        return DepthBoxFittingRepository(session).list_lidar_pointclouds_by_run(
            params_id
        )


def load_box_fittings(
    params_id: str,
    *,
    sample_data_tokens: list[str] | None = None,
    include_points: bool = False,
    include_mask: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    with read_only_session() as session:
        return DepthBoxFittingRepository(session).list_box_fittings_by_run(
            params_id,
            sample_data_tokens=sample_data_tokens,
            include_points=include_points,
            include_mask=include_mask,
        )


# ── 再フィルタ（パラメータ調整用）────────────────────────────────────────────

def refilter_sample(
    dataset_id: str,
    scene_token: str,
    params_id: str,
    sample_token: str,
    *,
    depth_params: dict[str, Any],
    lidar_params: dict[str, Any] | None = None,
    channels: list[str] | None = None,
) -> dict[str, Any]:
    """1 sample のインスタンス点群を、新しいパラメータで作り直す.

    保存済みの深度マップとクロージング後マスクから再生成するため、
    パイプライン全体を回し直す必要がない。

    Args:
        channels: 対象カメラ。None なら全カメラ。
            表示中のカメラだけに絞ると応答が速くなる

    Returns:
        {instance_tracking_2d_id ではなく BoxFitting3D.id: 結果} の dict。
        結果は num_points_raw / num_points_kept / points_raw_ego /
        points_filtered_ego を持つ。

    NOTE: 結果は DB に書かない。run の記録は「実行時のパラメータで得た点群」
    であるべきで、調整中の値で上書きすると来歴が壊れる。
    調整後の値で本番 run を回せば、その値が depth_params として記録される。
    """
    from app.services.inference_client import refilter_boxfitting

    settings = get_settings()

    with read_only_session() as session:
        repo = DepthBoxFittingRepository(session)
        depth_info = repo.list_depth_estimations_by_run(params_id)
        frames = SensorRepository(session).list_frames_by_sample(
            sample_token, keyframe_only=True
        )
        tokens = [
            f["token"] for f in frames
            if f["modality"] == "camera"
            and (channels is None or f["channel"] in channels)
        ]
        fittings = repo.list_box_fittings_by_run(
            params_id, sample_data_tokens=tokens, include_mask=True
        )

    frame_by_token = {f["token"]: f for f in frames}
    payload_frames = []
    for token, items in fittings.items():
        info = depth_info.get(token)
        frame = frame_by_token.get(token)
        if not info or not info.get("depth_path") or frame is None:
            continue
        instances = [
            {
                "id": fit["id"],
                "track_id": fit.get("track_id"),
                "label": fit.get("label"),
                "mask_rle_closed": fit["mask_rle_closed"],
            }
            for fit in items if fit.get("mask_rle_closed")
        ]
        if not instances:
            continue
        payload_frames.append({
            "depth_path": info["depth_path"],
            "calibrated_sensor": frame["calibrated_sensor"],
            "width": frame["width"],
            "height": frame["height"],
            "instances": instances,
        })

    if not payload_frames:
        return {}

    response = refilter_boxfitting({
        "frames": payload_frames,
        "depth_params": depth_params,
        "lidar_params": lidar_params or {},
        "stored_points_max": settings.BOXFIT_STORED_POINTS_MAX,
    })
    logger.info(
        "refiltered %d instances in %.2fs",
        len(response.get("instances", [])), response.get("elapsed_sec", 0.0),
    )
    return {item["id"]: item for item in response.get("instances", [])}


# ── 削除 ──────────────────────────────────────────────────────────────────────

def delete_run(params_id: str) -> None:
    """run を削除する（DB 行と派生ファイルの両方）.

    ファイルを消し忘れると DERIVED_ROOT が肥大し続ける。
    1 run で深度マップ約 140MB + LiDAR 約 140MB あるため影響が大きい。
    """
    _remove_derived_files(params_id)
    with session_scope() as session:
        DepthBoxFittingRepository(session).delete_run(params_id)


def prune_runs(dataset_id: str, scene_token: str, keep: int | None = None) -> list[str]:
    """古い run を削除する（派生ファイルも消す）."""
    settings = get_settings()
    keep = settings.DEPTH_MAX_RUNS_PER_SCENE if keep is None else keep
    with session_scope() as session:
        targets = DepthBoxFittingRepository(session).prune_runs(
            dataset_id, scene_token, keep=keep
        )
    for params_id in targets:
        _remove_derived_files(params_id)
    return targets


def _remove_derived_files(params_id: str) -> None:
    directory = run_derived_dir(params_id)
    if not directory.exists():
        return
    try:
        shutil.rmtree(directory)
        logger.info("removed derived files: %s", directory)
    except OSError as exc:  # noqa: BLE001
        # ファイルが消せなくても DB 側の整合は保ちたいので、警告に留める
        logger.warning("failed to remove derived files %s: %s", directory, exc)
