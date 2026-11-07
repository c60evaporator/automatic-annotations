"""Orient-Anything-V2（物体の向き推定）のラッパー（公式リポジトリ版）.

公式リポジトリ（https://github.com/SpatialVision/Orient-Anything-V2）は
パッケージになっておらず、ルート直下の `vision_tower` / `vggt` / `utils` を
直接 import する作り。`utils` / `vggt` は一般的な名前なので PYTHONPATH には入れず、
ロード時にだけ sys.path へ足す。

`utils/app_utils.py` は先頭で rembg（背景除去）を import するが、使うのは
背景除去のときだけ。背景はトラッキングのマスクで落とせるため rembg は入れず、
空のモジュールを差し込んで import を通す。

重みは Hugging Face（Viglong/OriAnyV2_ckpt、約 5GB）から HF_HOME へダウンロードする。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# 出力のビン構成（公式の inf_single_batch と同じ）。
# 方位角 360（0..359 度）、仰角 180（-90..89 度）、回転 360（-180..179 度）
AZIMUTH_BINS = 360
ELEVATION_BINS = 180
# 推論に使わないヘッド（VGGT の頭部）。読み込み後に外して VRAM を空ける
UNUSED_HEADS = ("camera_head", "point_head", "depth_head", "track_head")


def _import_official(root: Path):
    """公式実装を import する（モデルクラスと前処理・対称性の推定関数）."""
    if "rembg" not in sys.modules:
        try:
            import rembg  # noqa: F401
        except ImportError:
            sys.modules["rembg"] = types.ModuleType("rembg")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from utils.app_utils import preprocess_images, val_fit_alpha
    from vision_tower import VGGT_OriAny_Ref

    return VGGT_OriAny_Ref, preprocess_images, val_fit_alpha


def crop_object_on_white(
    image: np.ndarray,
    bbox: list[float],
    mask: np.ndarray | None = None,
    *,
    foreground_ratio: float = 0.85,
) -> np.ndarray:
    """物体を白背景の正方形に切り出す（公式の前処理に合わせる）.

    公式は rembg で背景を消し、物体が長辺の 85% を占めるよう白で余白を足す
    （background_preprocess → resize_foreground(0.85)）。学習データも白背景の
    レンダリングなので、背景を残すと向きが崩れる。rembg の代わりに
    トラッキングのマスクで背景を白にする。

    Args:
        mask: 画像と同じ解像度の bool マスク。None なら背景を残す
    """
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, width), min(y1, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"空の外接矩形です: {bbox}")
    crop = image[y0:y1, x0:x1].copy()
    if mask is not None:
        crop[~mask[y0:y1, x0:x1]] = 255
    side = int(np.ceil(max(crop.shape[:2]) / foreground_ratio))
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    top = (side - crop.shape[0]) // 2
    left = (side - crop.shape[1]) // 2
    canvas[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return canvas


def download_checkpoint(repo_id: str, filename: str, revision: str | None) -> Path:
    """重みを HF_HOME へダウンロードする（済んでいればキャッシュのパスを返す）."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))


class OrientAnythingV2Estimator:
    """物体のクロップ画像から向き（方位角・仰角・回転）を推定する."""

    def __init__(self, checkpoint_path: Path, root: Path, device: str):
        import torch

        model_cls, self._preprocess, self._fit_alpha = _import_official(root)
        on_cuda = device.startswith("cuda") and torch.cuda.is_available()
        # 公式と同じ。Ampere 以降は bf16
        dtype = (torch.bfloat16 if on_cuda and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)
        model = model_cls(out_dim=900, dtype=dtype, nopretrain=True)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        del state_dict
        for name in UNUSED_HEADS:
            setattr(model.vggt, name, None)
        self.model = model.eval().to(device)
        self.device = device

    def to(self, device: str) -> "OrientAnythingV2Estimator":
        self.model.to(device)
        self.device = device
        return self

    def predict(self, crops: list[np.ndarray], batch_size: int = 8) -> dict[str, np.ndarray]:
        """RGB のクロップ群から向きを推定する.

        公式と同じく、最大辺を 518px にして白で 518x518 へ埋める（pad）。
        背景を白にしたい場合は、呼び出し側でマスク外を白にしてから渡す。

        Returns:
            azimuth_deg / elevation_deg / rotation_deg: (N,) 度
            alpha: (N,) 対称性（0: 向きが定まらない、1: 一意、2: 前後対称、4: 4 方向対称）
            confidence: (N,) 方位角の argmax ビンの確率（sigmoid）
        """
        import torch
        from PIL import Image

        keys = ("azimuth_deg", "elevation_deg", "rotation_deg", "alpha", "confidence")
        if not crops:
            return {key: np.zeros(0, dtype=np.float32) for key in keys}
        out: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                images = [Image.fromarray(np.asarray(c)) for c in crops[start:start + batch_size]]
                batch = self._preprocess(images, mode="pad").to(self.device)
                pose_enc = self.model(batch[:, None]).reshape(len(images), -1).float()
                az_logits = pose_enc[:, :AZIMUTH_BINS]
                el_logits = pose_enc[:, AZIMUTH_BINS:AZIMUTH_BINS + ELEVATION_BINS]
                ro_logits = pose_enc[:, AZIMUTH_BINS + ELEVATION_BINS:]
                # 公式は方位角を BCE で学習しているので sigmoid が確率になる
                az_probs = torch.sigmoid(az_logits).cpu().numpy()
                out["azimuth_deg"].append(az_logits.argmax(-1).cpu().numpy().astype(np.float32))
                out["elevation_deg"].append(
                    (el_logits.argmax(-1) - 90).cpu().numpy().astype(np.float32))
                out["rotation_deg"].append(
                    (ro_logits.argmax(-1) - 180).cpu().numpy().astype(np.float32))
                out["alpha"].append(np.asarray(self._fit_alpha(distribute=az_probs),
                                               dtype=np.float32))
                out["confidence"].append(az_probs.max(axis=1).astype(np.float32))
        return {key: np.concatenate(values) for key, values in out.items()}
