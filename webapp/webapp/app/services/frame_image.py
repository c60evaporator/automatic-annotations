"""センサー画像の取得（Streamlit キャッシュ付き）.

basemap と同じ2段構成:
  1. list_frames_by_sample() … @st.cache_data / DB から filename を引く
  2. load_image_cached()     … @st.cache_resource / PIL 画像本体を保持

画像の実体は filename で一意なので、キャッシュキーも filename にする。
sample_token + sensor_token をキーにすると、同じ画像を別の切り口で
参照したときに二重に載る。

max_entries を付けているのは、カメラ画像が 1600x900 で1枚あたり
数MB のメモリを占め、シーン全体を舐めると青天井に増えるため。
LRU で古いものから捨てられる。
"""
from __future__ import annotations

from typing import Any

import streamlit as st
from PIL import Image

from app.services.data_file_service import load_image
from app.streamlit.data_access import list_frames_by_sample

# 保持する画像の枚数。6カメラ × 十数フレーム分を想定
MAX_CACHED_IMAGES = 64


@st.cache_resource(max_entries=MAX_CACHED_IMAGES, show_spinner=False)
def load_image_cached(
    dataroot: str, filename: str, max_size: tuple[int, int] | None = None
) -> Image.Image | None:
    """画像を読み込んでプロセス内で共有する.

    PIL の Image は @st.cache_data のシリアライズに向かないため
    cache_resource を使う。返す画像は読み取り専用として扱うこと
    （呼び出し側で加工する場合は .copy() してから）。
    """
    return load_image(dataroot, filename, max_size=max_size)


def get_keyframe_image(
    dataset_id: str,
    dataroot: str,
    sample_token: str,
    sensor_token: str,
    *,
    max_size: tuple[int, int] | None = None,
) -> tuple[Image.Image | None, dict[str, Any] | None]:
    """指定 sample / センサーのキーフレーム画像を返す.

    Returns:
        (画像, フレーム情報)。該当フレームが無ければ (None, None)、
        ファイルが見つからなければ (None, frame)。
        フレーム情報には channel / calibrated_sensor / ego_pose が含まれるので、
        投影処理をそのまま続けられる。
    """
    frames = list_frames_by_sample(
        dataset_id, sample_token, keyframe_only=True, sensor_token=sensor_token
    )
    if not frames:
        return None, None

    frame = frames[0]
    if frame["modality"] != "camera":
        # LiDAR/RADAR の sample_data を渡された場合は画像として読まない
        return None, frame

    image = load_image_cached(dataroot, frame["filename"], max_size)
    return image, frame


def get_frame_images_by_scene(
    dataset_id: str,
    dataroot: str,
    frames: list[dict[str, Any]],
    *,
    max_size: tuple[int, int] | None = None,
) -> list[tuple[Image.Image | None, dict[str, Any]]]:
    """list_frames_by_scene() の結果に対して画像をまとめて解決する.

    フレーム一覧の取得は呼び出し側で1回だけ行い、ここでは画像だけを付ける。
    枚数が MAX_CACHED_IMAGES を超えるとキャッシュから溢れるため、
    シーン全体を一度に読むのではなく表示範囲を絞って渡すこと。
    """
    out: list[tuple[Image.Image | None, dict[str, Any]]] = []
    for frame in frames:
        if frame["modality"] != "camera":
            out.append((None, frame))
            continue
        out.append((load_image_cached(dataroot, frame["filename"], max_size), frame))
    return out
