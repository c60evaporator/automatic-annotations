"""トラッキング入力フレームの選択（sweep）.

SAM2 にはキーフレームだけでなく非キーフレーム（sweep）も与えて伝播させる。
どのフレームを使うかは webapp 側で決めて推論サーバーへ渡す
（推論サーバーは DB を持たないため）。

選択方法は nuScenes devkit ベースの実装に合わせた "uniform"。
sample 内のフレームをキーフレームから逆算して等間隔に選ぶ。

  - キーフレームは必ず含まれる（sample 内で最も新しいフレームがキーフレーム）
  - 区間 [0, len) の半開区間からオフセットを取るため、
    sample の境界をまたいでも間隔がほぼ均等に保たれる
    （index -1 は前の sample のキーフレームなので上端を含めない）
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

# 1 sample あたりの sweep 数の上限（UI の number_input の上限に使う）
MAX_SWEEPS_PER_SAMPLE = 10


def select_uniform_sweep_indices(num_frames: int, num_sweeps: int) -> list[int]:
    """sample 内フレームから使うものの index を返す.

    Args:
        num_frames: その sample・そのセンサーのフレーム数（キーフレーム含む）
        num_sweeps: 使いたいフレーム数（1 ならキーフレームのみ）

    Returns:
        昇順の index。キーフレーム（末尾 = num_frames-1）を必ず含む。

    例: num_frames=6 のとき
        num_sweeps=2 -> [2, 5]
        num_sweeps=3 -> [1, 3, 5]
    """
    if num_frames <= 0:
        return []
    if num_sweeps < 1:
        raise ValueError("num_sweeps must be at least 1")

    # numpy.linspace(0, num_frames, num_sweeps, endpoint=False) と同値。
    # webapp 側の依存を増やさないため手計算にしてある
    offsets = [i * num_frames / num_sweeps for i in range(num_sweeps)]
    indices = {num_frames - 1 - int(offset) for offset in offsets}
    # 範囲外（num_sweeps > num_frames のとき負になる）は捨てる
    return sorted(i for i in indices if 0 <= i < num_frames)


def select_tracking_frames(
    frames: list[dict[str, Any]], num_sweeps: int
) -> list[dict[str, Any]]:
    """シーン全体のフレームから、トラッキングに使うものを選ぶ.

    Args:
        frames: SensorRepository.list_frames_by_scene(keyframe_only=False) の結果。
            カメラ以外も混ざっていてよい（呼び出し側で絞っても可）
        num_sweeps: 1 sample あたりのフレーム数

    Returns:
        選ばれたフレーム。(channel, sample の時刻) 順。

    sweep は「センサーごと・sample ごと」に記録されるため、
    グルーピングは (channel, sample_token) で行う。
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        if frame.get("modality") != "camera":
            continue
        grouped[(frame["channel"], frame["sample_token"])].append(frame)

    selected: list[dict[str, Any]] = []
    for (_channel, _sample_token), group in grouped.items():
        group.sort(key=lambda f: f["timestamp"])
        for index in select_uniform_sweep_indices(len(group), num_sweeps):
            selected.append(group[index])

    selected.sort(key=lambda f: (f["channel"], f["timestamp"]))
    return selected


def count_frames_per_sample(frames: list[dict[str, Any]]) -> dict[str, int]:
    """(channel, sample) ごとのフレーム数を返す（UI の上限提示用）.

    データセットによって sweep の本数が違うため、
    「指定した num_sweeps が実際に取れるか」を UI で示すのに使う。
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for frame in frames:
        if frame.get("modality") != "camera":
            continue
        counts[(frame["channel"], frame["sample_token"])] += 1
    per_channel: dict[str, int] = {}
    for (channel, _sample), n in counts.items():
        # 最小値を採用する。sample によって本数が違う場合、
        # 少ないほうに合わせないと指定どおりの sweep 数が取れない
        per_channel[channel] = min(per_channel.get(channel, n), n)
    return per_channel
