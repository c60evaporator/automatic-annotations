"""SAM2 Video Predictor の低レベル操作.

公式リポジトリ（facebookresearch/sam2）の `SAM2VideoPredictor` を、
ディレクトリではなく **PIL 画像のリスト**から使えるようにする。
`init_state()` は動画ファイル / JPEG フォルダを前提にしているが、
こちらは DB から選んだ任意のフレーム列（sweep を間引いたもの）を
そのまま渡したいため、状態を自前で組み立てる。

torch / sam2 への import はこのモジュール内に閉じてあり、
モデルの初回ロード時にだけ読み込まれる。
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from app.core.logging import get_logger

logger = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _autocast(device: str):
    """CUDA のときだけ bfloat16 の autocast を有効にする."""
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _load_video_frames(
    images: list[Image.Image],
    image_size: int,
    compute_device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """PIL 画像のリストを SAM2 入力用のテンソルへ変換する.

    Returns:
        (画像テンソル ``(N, C, H, W)``, 元の高さ, 元の幅)
    """
    to_tensor = v2.Compose([v2.ToImage()])
    tensors = torch.stack([to_tensor(image) for image in images])
    original_height, original_width = tensors.shape[-2], tensors.shape[-1]

    resize = v2.Compose([v2.Resize(size=(image_size, image_size), antialias=True)])
    normalize = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensors = resize(tensors)
    # pin_memory + non_blocking で CPU→GPU 転送を重ねる
    if compute_device.type == "cuda":
        tensors = tensors.pin_memory()
    tensors = tensors.to(compute_device, non_blocking=True)
    tensors = normalize(tensors)

    return tensors, original_height, original_width


def init_frame_state(predictor: Any, images: list[Image.Image]) -> dict:
    """PIL 画像のリストから inference_state を組み立てる.

    `predictor.init_state()` の代わり。オフロードは行わない
    （フレーム数が 1 区間ぶん = 十数枚と少ないため、
      CPU へ退避するより GPU に置いたままのほうが速い）。
    """
    image_tensors, video_height, video_width = _load_video_frames(
        images=images,
        image_size=predictor.image_size,
        compute_device=predictor.device,
    )

    inference_state: dict[str, Any] = {}
    inference_state["images"] = image_tensors
    inference_state["num_frames"] = len(images)
    inference_state["offload_video_to_cpu"] = False
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = predictor.device
    inference_state["storage_device"] = predictor.device
    # 各フレームへの入力
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # 直近フレームの特徴キャッシュ
    inference_state["cached_features"] = {}
    # フレーム間で変わらない値
    inference_state["constants"] = {}
    # クライアント側 object id とモデル側 index の対応
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # 先頭フレームの特徴を先に計算してウォームアップする
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


def _logits_to_masks(mask_logits: Any) -> np.ndarray:
    """SAM2 の出力 logits ``(N, 1, H, W)`` を bool マスク ``(N, H, W)`` にする.

    NOTE: np.squeeze(tensor, axis=1) は使わないこと。
    numpy が torch テンソルの squeeze へ委譲する挙動は
    引数名（axis / dim）の違いで壊れやすい。torch 側の演算で完結させる。
    """
    masks = (mask_logits.squeeze(1) > 0.0)
    return masks.cpu().numpy()


def add_box_prompts(
    predictor: Any,
    inference_state: dict,
    frame_idx: int,
    box_prompts: list[dict[str, Any]],
) -> tuple[np.ndarray, list[int]]:
    """ボックスプロンプトを追加し、そのフレームのマスクを返す.

    Args:
        box_prompts: ``{"xmin","ymin","xmax","ymax","local_id"}`` のリスト。
            local_id が SAM2 の obj_id になる

    Returns:
        (マスク配列 ``(N, H, W)`` の bool, obj_id のリスト)
    """
    out_obj_ids: list[int] = []
    out_mask_logits = None

    for box in box_prompts:
        box_array = np.array(
            [box["xmin"], box["ymin"], box["xmax"], box["ymax"]], dtype=np.float32
        )
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=int(box["local_id"]),
            points=None,
            labels=None,
            box=box_array,
        )

    if out_mask_logits is None:
        return np.zeros((0, 1, 1), dtype=bool), []

    return _logits_to_masks(out_mask_logits), list(out_obj_ids)


def propagate_inference(
    predictor: Any,
    inference_state: dict,
    start_frame_idx: int = 0,
) -> dict[int, tuple[np.ndarray, list[int]]]:
    """全フレームへ伝播させ、フレームごとのマスクを返す.

    Returns:
        ``{frame_idx: (マスク (N, H, W), obj_id のリスト)}``
    """
    results: dict[int, tuple[np.ndarray, list[int]]] = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=start_frame_idx
    ):
        results[frame_idx] = (_logits_to_masks(mask_logits), list(obj_ids))
    return results
