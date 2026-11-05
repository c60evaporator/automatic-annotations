"""複数フレーム深度推定のスライディングウィンドウ."""
from __future__ import annotations


def create_sliding_windows(
    num_frames: int, window_size: int, stride: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """フレーム列をウィンドウへ分割し、各ウィンドウで結果を採用する範囲を返す.

    結果はウィンドウの中央寄りから採用する（端のフレームは片側の文脈しか無いため）。
    各フレームはちょうど 1 つのウィンドウで採用される。

    Returns:
        (window_ranges, used_ranges)
        window_ranges: ウィンドウの範囲 ``(start, end)``（全体の index、end は含まない）
        used_ranges: 結果を採用する範囲 ``(start, end)``（ウィンドウ内の相対 index）
    """
    if window_size < 1 or not 1 <= stride <= window_size:
        raise ValueError(
            f"window_size は 1 以上、stride は 1 以上 window_size 以下にしてください: "
            f"window_size={window_size}, stride={stride}"
        )
    if num_frames <= 0:
        return [], []
    if num_frames <= window_size:
        return [(0, num_frames)], [(0, num_frames)]

    last_start = num_frames - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)

    left_context = (window_size - stride) // 2
    window_ranges: list[tuple[int, int]] = []
    used_ranges: list[tuple[int, int]] = []
    used_start = 0
    for index, start in enumerate(starts):
        is_last = index == len(starts) - 1
        used_end = num_frames if is_last else start + left_context + stride
        window_ranges.append((start, start + window_size))
        used_ranges.append((used_start - start, used_end - start))
        used_start = used_end
    return window_ranges, used_ranges
