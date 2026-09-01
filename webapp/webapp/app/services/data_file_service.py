"""センサーデータ（画像・点群）のファイル読み込み.

DB には filename（dataroot からの相対パス）しか持たず、実体は
マウントしたデータセットフォルダから直接読む。その解決と読み込みを担う。

この層に置く理由:
  - Streamlit に依存しない（推論サーバーへの入力作成やエクスポートでも使う）
  - DB にも依存しない（filename を受け取るだけ）
  パス解決は「どのデータセットの、どのファイルか」が決まれば済む話であり、
  sample_token → filename の解決は Repository / data_access の責務。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 想定するカメラ画像の拡張子
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DataPathError(ValueError):
    """データセット外を指すパスが渡された."""


def resolve_path(dataroot: str, filename: str) -> Path:
    """dataroot 相対の filename を実パスに解決する.

    DATA_ROOT/<dataroot> の外に出るパスは拒否する。
    filename は DB 由来だが、インポート元 JSON の内容がそのまま入るため
    信頼せずに検証する（`../` を含む値が混ざると読み取り範囲が広がる）。
    """
    settings = get_settings()
    root = (settings.DATA_ROOT / dataroot).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise DataPathError(f"データセット外のパスです: {filename}")
    return path


def load_image(
    dataroot: str,
    filename: str,
    *,
    max_size: tuple[int, int] | None = None,
) -> Image.Image | None:
    """カメラ画像を PIL.Image として読み込む.

    Args:
        dataroot: Dataset.dataroot（DATA_ROOT からの相対パス）
        filename: SampleData.filename（dataroot からの相対パス）
        max_size: 指定すると縦横がこれに収まるよう縮小する（表示用）

    Returns:
        画像。存在しない場合は None（画面を落とさないため例外にしない）
    """
    path = resolve_path(dataroot, filename)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        logger.warning("not an image file: %s", filename)
        return None
    if not path.exists():
        logger.warning("image not found: %s", path)
        return None

    with Image.open(path) as img:
        img.load()  # with を抜けた後も使えるよう、ここで実体を読む
        image = img.convert("RGB") if img.mode not in ("RGB", "L") else img.copy()

    if max_size is not None:
        # thumbnail は縦横比を保ったまま縮小する（破壊的なので copy 済みに適用）
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def exists(dataroot: str, filename: str) -> bool:
    """ファイルの実体があるかを確認する（一覧表示のチェック用）."""
    try:
        return resolve_path(dataroot, filename).exists()
    except DataPathError:
        return False
