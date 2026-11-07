"""PartHOE（人物の水平向き推定）のラッパー（公式リポジトリ版）.

公式リポジトリ（https://github.com/zhaojieting/Part_HOE）の
`tensorrt/nn_utils.py` にある `MinimalPartHOE` を使う。torch だけで動き、
`lib/` 側（timm / yacs / tensorboardX 等が必要）は使わない。
`nn_utils` はパッケージではなく、名前も一般的で衝突しやすいため、
ファイルパスから別名でロードする。

重みは Hugging Face に無いため、初回ロード時に Google Drive から
ダウンロードする（`ensure_checkpoint`）。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# 入力サイズ (H, W)。公式の config/parthoe.yaml の IMAGE_SIZE [192, 256]（W, H）に対応
INPUT_SIZE = (256, 192)
# 出力は 72 ビン（5 度刻み）の確率
NUM_BINS = 72
BIN_DEG = 360.0 / NUM_BINS
# 公式の推論コードと同じ（ImageNet の平均・標準偏差）
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_checkpoint(path: Path, url: str) -> Path:
    """重みが無ければダウンロードする.

    途中で落ちても壊れたファイルが残らないよう、一時ファイルに書いてから置き換える。
    Google Drive は失敗時に HTML（確認ページ等）を返すため、それを弾く。
    """
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    logger.info("downloading PartHOE checkpoint to %s", path)
    with urllib.request.urlopen(url, timeout=60) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if content_type.startswith("text/html"):
            raise RuntimeError(
                f"PartHOE の重みを取得できませんでした（HTML が返りました）: {url}\n"
                f"手動でダウンロードして {path} に置いてください"
            )
        with open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
    os.replace(tmp, path)
    return path


def crop_person(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    """画像から人物を切り出して、モデルの入力サイズ (H, W) にする.

    公式の学習データ（coco_hoe_dataset._box2cs）と同じく、外接矩形を
    入力のアスペクト比（3:4）まで広げ、さらに 1.25 倍の余白を取る。
    縦長の矩形をそのまま引き伸ばすと人物が歪み、向きを取り違える。
    はみ出した範囲は黒で埋める（公式と同じ）。
    """
    import cv2

    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    w, h = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    out_h, out_w = INPUT_SIZE
    aspect = out_w / out_h
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    scale = out_w / (w * 1.25)
    matrix = np.array([
        [scale, 0.0, out_w / 2.0 - scale * cx],
        [0.0, scale, out_h / 2.0 - scale * cy],
    ], dtype=np.float32)
    return cv2.warpAffine(
        image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _load_minimal_parthoe_class(parthoe_root: Path):
    src = parthoe_root / "tensorrt" / "nn_utils.py"
    spec = importlib.util.spec_from_file_location("parthoe_nn_utils", src)
    if spec is None or spec.loader is None:
        raise ImportError(f"PartHOE の実装が見つかりません: {src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MinimalPartHOE


class PartHOEEstimator:
    """人物クロップ画像から水平向き（72 ビンの確率）を推定する."""

    def __init__(self, checkpoint_path: Path, parthoe_root: Path, device: str):
        import torch

        model_cls = _load_minimal_parthoe_class(parthoe_root)
        # PartHOE-S（ViT-S）の構成。公式の tensorrt/quantize_parthoe.py と同じ
        model = model_cls(
            img_size=INPUT_SIZE, patch_size=16, embed_dim=384, depth=12, num_heads=12
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        self.model = model.eval().to(device)
        self.device = device

    def to(self, device: str) -> "PartHOEEstimator":
        self.model.to(device)
        self.device = device
        return self

    def preprocess(self, crops: list[np.ndarray]) -> np.ndarray:
        """RGB uint8 のクロップ群を (N, 3, H, W) の正規化済み配列にする."""
        import cv2

        h, w = INPUT_SIZE
        batch = np.stack([
            cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR) for crop in crops
        ]).astype(np.float32) / 255.0
        batch = (batch - _MEAN) / _STD
        return batch.transpose(0, 3, 1, 2)

    def predict(self, crops: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """各クロップの向きの確率 (N, 72) を返す. ビン i は i * 5 度."""
        import torch

        if not crops:
            return np.zeros((0, NUM_BINS), dtype=np.float32)
        outputs = []
        with torch.inference_mode():
            for i in range(0, len(crops), batch_size):
                x = torch.from_numpy(self.preprocess(crops[i:i + batch_size])).to(self.device)
                _, hoe, _ = self.model(x)
                outputs.append(hoe.float().cpu().numpy())
        return np.concatenate(outputs)

    @staticmethod
    def to_degrees(probs: np.ndarray) -> np.ndarray:
        """確率の argmax を角度 [deg] にする（公式の評価コードと同じ）."""
        return probs.argmax(axis=1) * BIN_DEG
