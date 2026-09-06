"""カメラの表示順.

DB から取得したセンサー一覧は channel 名の昇順（CAM_BACK が先頭）で返るが、
確認したいのは前方カメラであることが多い。表示順を設定から与える。

順序は「2 列グリッドに並べたときの見え方」を決める点に注意。
先頭 2 つが 1 行目、次の 2 つが 2 行目…と配置される。
"""
from __future__ import annotations

from typing import Any, Sequence

from app.core.config import get_settings


def order_cameras(
    sensors: Sequence[dict[str, Any]], order: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """設定の順序でカメラを並べ替える.

    Args:
        sensors: SensorRepository.list_sensors() の結果
        order: channel 名の並び。None なら Settings.CAM_DISPLAY_ORDER

    Returns:
        並べ替えたリスト。

    設定に無い channel は末尾へ回し、その中では channel 名順にする。
    データセットによってカメラ構成が違っても、設定を書き換えずに動く
    （並びが不定になるより、既知のものを優先して残りを安定順で出すほうがよい）。
    """
    if order is None:
        order = get_settings().CAM_DISPLAY_ORDER

    rank = {channel: index for index, channel in enumerate(order)}
    unknown_base = len(rank)

    def sort_key(sensor: dict[str, Any]) -> tuple[int, str]:
        channel = sensor.get("channel", "")
        return (rank.get(channel, unknown_base), channel)

    return sorted(sensors, key=sort_key)
