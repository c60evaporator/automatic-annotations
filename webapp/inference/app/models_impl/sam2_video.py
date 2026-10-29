"""SAM2 Video（Transformers 版）の低レベル操作.

Transformers の `Sam2VideoModel` / `Sam2VideoProcessor` を、
DB から選んだ任意のフレーム列（sweep を間引いたもの）の **PIL 画像のリスト**
から使えるようにする。

torch / transformers への import はこのモジュール内に閉じてあり、
モデルの初回ロード時にだけ読み込まれる。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


def init_frame_state(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    device: str,
    dtype: torch.dtype,
) -> Any:
    """PIL 画像のリストから推論セッション（inference_state）を作る.

    フレームは前処理して推論デバイスに置く（1 区間ぶん = 十数枚と少ないため、
    CPU へ退避するより GPU に置いたままのほうが速い）。

    Returns:
        Transformers の Sam2VideoInferenceSession
    """
    return processor.init_video_session(
        video=[image.convert("RGB") for image in images],
        inference_device=device,
        dtype=dtype,
    )


def _logits_to_masks(processor: Any, inference_state: Any, pred_masks: torch.Tensor) -> np.ndarray:
    """低解像度の mask logits を元画像サイズへ戻し、bool マスク ``(N, H, W)`` にする."""
    original_size = [int(inference_state.video_height), int(inference_state.video_width)]
    mask_logits = processor.post_process_masks(
        [pred_masks], original_sizes=[original_size], binarize=False
    )[0]  # (N, 1, H, W)
    return (mask_logits[:, 0] > 0.0).cpu().numpy()


def add_box_prompts(
    model: Any,
    processor: Any,
    inference_state: Any,
    frame_idx: int,
    box_prompts: list[dict[str, Any]],
) -> tuple[np.ndarray, list[int]]:
    """ボックスプロンプトを追加し、そのフレームのマスクを返す.

    区間ごとに新しいセッションを作るので、プロンプトを与えるフレームは
    どのオブジェクトにとっても初回の条件フレームになる。全ボックスを
    1 回でまとめて追加し、1 回の推論で全オブジェクトのマスクを得る。

    Args:
        box_prompts: ``{"xmin","ymin","xmax","ymax","local_id"}`` のリスト。
            local_id が SAM2 の obj_id になる

    Returns:
        (マスク配列 ``(N, H, W)`` の bool, obj_id のリスト)
    """
    if not box_prompts:
        return np.zeros((0, 1, 1), dtype=bool), []

    processor.add_inputs_to_inference_session(
        inference_session=inference_state,
        frame_idx=frame_idx,
        obj_ids=[int(box["local_id"]) for box in box_prompts],
        # [image, box, xyxy] の入れ子形式
        input_boxes=[[[float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])]
                      for box in box_prompts]],
    )
    outputs = model(inference_session=inference_state, frame_idx=frame_idx)
    return _logits_to_masks(processor, inference_state, outputs.pred_masks), list(outputs.object_ids)


def propagate_inference(
    model: Any,
    processor: Any,
    inference_state: Any,
    start_frame_idx: int = 0,
    reverse: bool = False,
) -> dict[int, tuple[np.ndarray, list[int]]]:
    """プロンプトを与えたフレームから伝播させ、フレームごとのマスクを返す.

    Args:
        start_frame_idx: 伝播の起点。backward ではプロンプトを与えた
            **最終フレームの index** を渡す
        reverse: True で時間をさかのぼって伝播する

    Returns:
        ``{frame_idx: (マスク (N, H, W), obj_id のリスト)}``

    NOTE: reverse=True でも frame_idx は区間内の絶対 index が返る。
    呼び出し側でフレーム順に並べ替える必要はない。
    """
    results: dict[int, tuple[np.ndarray, list[int]]] = {}
    for outputs in model.propagate_in_video_iterator(
        inference_state, start_frame_idx=start_frame_idx, reverse=reverse
    ):
        results[outputs.frame_idx] = (
            _logits_to_masks(processor, inference_state, outputs.pred_masks),
            list(outputs.object_ids),
        )
    return results
