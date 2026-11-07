"""Orient-Anything（V1、物体の向き推定）のラッパー.

公式リポジトリ（https://github.com/SpatialVision/Orient-Anything）のモデルは
DINOv2 の CLS トークンに 2 層の MLP を付けただけの構造（`vision_tower.DINOv2_MLP`）。
公式のコードはデモ用の依存（rembg・描画）を import するため使わず、
同じ構造を transformers の `Dinov2Model` で組んで重みだけを読み込む。

- 重み: Hugging Face（Viglong/Orient-Anything の `croplargeEX2/dino_weight.pt`、
  DINOv2-Large ベース、約 3 億パラメータ・1.2GB）。バックボーンの重みも含む
- バックボーンの構成と前処理: Hugging Face の facebook/dinov2-large
  （config.json / preprocessor_config.json だけ使い、重みは落とさない）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# 出力のビン構成（公式の get_3angle と同じ）。
# 方位角 360（0..359 度）、仰角 180（-90..89 度）、回転 180（-180..-1 度。公式どおり）、
# 末尾 2 つが「向きを持つ物体か」の 2 クラス
AZIMUTH_BINS = 360
ELEVATION_BINS = 180
ROTATION_BINS = 180
CONFIDENCE_BINS = 2
OUTPUT_DIM = AZIMUTH_BINS + ELEVATION_BINS + ROTATION_BINS + CONFIDENCE_BINS
# 新しい transformers の Dinov2 は注意層の名前が変わっている（旧名 → 新名）
_RENAMED_KEYS = {
    "attention.attention.query": "attention.q_proj",
    "attention.attention.key": "attention.k_proj",
    "attention.attention.value": "attention.v_proj",
    "attention.output.dense": "attention.o_proj",
}


def crop_object_on_white(
    image: np.ndarray,
    bbox: list[float],
    mask: np.ndarray | None = None,
    *,
    foreground_ratio: float = 0.85,
) -> np.ndarray:
    """物体を白背景の正方形に切り出す（公式の前処理に合わせる）.

    公式は rembg で背景を消し、物体が長辺の 85% を占めるよう余白を足す
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


def _build_model(backbone: str, backbone_revision: str | None):
    """公式の DINOv2_MLP と同じ構造のモデルを、重み無しで組む."""
    import torch
    from torch import nn
    from transformers import Dinov2Config, Dinov2Model

    class MLPHead(nn.Module):
        # 公式の MLP_dim と同じ（Linear + BN + ReLU → Linear + BN）
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.net1 = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                                      nn.ReLU(inplace=True))
            self.net2 = nn.Sequential(nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net2(self.net1(x))

    class OrientAnythingV1(nn.Module):
        # 公式の DINOv2_MLP（マスク無しの推論時）と同じ: CLS トークン → MLP
        def __init__(self):
            super().__init__()
            config = Dinov2Config.from_pretrained(backbone, revision=backbone_revision)
            self.dinov2 = Dinov2Model(config)
            self.down_sampler = MLPHead(config.hidden_size, OUTPUT_DIM)

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            hidden = self.dinov2(pixel_values=pixel_values).last_hidden_state
            return self.down_sampler(hidden[:, 0, :])

    return OrientAnythingV1()


def _load_state_dict(checkpoint_path: Path) -> dict:
    import torch

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    renamed = {}
    for key, value in state_dict.items():
        for old, new in _RENAMED_KEYS.items():
            key = key.replace(old, new)
        renamed[key] = value
    return renamed


class OrientAnythingV1Estimator:
    """物体のクロップ画像から向き（方位角・仰角・回転）を推定する."""

    def __init__(
        self,
        checkpoint_path: Path,
        backbone: str,
        device: str,
        *,
        backbone_revision: str | None = None,
    ):
        from transformers import AutoImageProcessor

        model = _build_model(backbone, backbone_revision)
        # 新旧どちらの名前でも読めるよう名前を揃えてから厳密に読む（取りこぼしを見逃さない）
        model.load_state_dict(_load_state_dict(checkpoint_path), strict=True)
        self.model = model.eval().to(device)
        # 公式と同じ前処理（短辺 256 にリサイズ → 中央 224 を切り出し → ImageNet 正規化）
        self.processor = AutoImageProcessor.from_pretrained(backbone, revision=backbone_revision)
        self.device = device

    def to(self, device: str) -> "OrientAnythingV1Estimator":
        self.model.to(device)
        self.device = device
        return self

    def predict(self, crops: list[np.ndarray], batch_size: int = 32) -> dict[str, np.ndarray]:
        """RGB のクロップ群から向きを推定する.

        背景を白にしたい場合は、呼び出し側でマスク外を白にしてから渡す
        （`crop_object_on_white`）。

        Returns:
            azimuth_deg / elevation_deg / rotation_deg: (N,) 度
            confidence: (N,) 向きを持つ物体である確率（公式の confidence）
        """
        import torch
        from PIL import Image

        keys = ("azimuth_deg", "elevation_deg", "rotation_deg", "confidence")
        if not crops:
            return {key: np.zeros(0, dtype=np.float32) for key in keys}
        out: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        el_start = AZIMUTH_BINS
        ro_start = el_start + ELEVATION_BINS
        conf_start = ro_start + ROTATION_BINS
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                images = [Image.fromarray(np.asarray(c)) for c in crops[start:start + batch_size]]
                inputs = self.processor(images=images, return_tensors="pt")
                pred = self.model(inputs["pixel_values"].to(self.device)).float()
                out["azimuth_deg"].append(
                    pred[:, :el_start].argmax(-1).cpu().numpy().astype(np.float32))
                out["elevation_deg"].append(
                    (pred[:, el_start:ro_start].argmax(-1) - 90).cpu().numpy().astype(np.float32))
                out["rotation_deg"].append(
                    (pred[:, ro_start:conf_start].argmax(-1) - 180).cpu().numpy().astype(np.float32))
                out["confidence"].append(
                    torch.softmax(pred[:, conf_start:], dim=-1)[:, 0].cpu().numpy().astype(np.float32))
        return {key: np.concatenate(values) for key, values in out.items()}
