"""basemap 画像のリサイズキャッシュ.

nuScenes の basemap PNG は 1 枚が数千万画素あり、そのまま Plotly に
渡すと描画が重い。0.1 倍程度に縮小したものをディスクにキャッシュする。

旧実装からの変更点:

  - 保存先を DERIVED_ROOT 配下に変更。
    データセットのマウントは読み取り専用（/data:ro）なので、
    データセットルート配下にキャッシュを書くことはできない。

  - 全マップを起動時に一括生成するのをやめ、必要になったものだけ作る。
    DB からシーンに対応する 1 枚を引けるようになったため、
    使わないマップまで縮小する必要がなくなった。

  - Streamlit に依存しない。進捗表示は呼び出し側の責務。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# basemap は非常に大きく、Pillow の decompression bomb 判定に引っかかる
Image.MAX_IMAGE_PIXELS = None

DEFAULT_SCALE = 0.1


def cache_path(dataset_id: str, basemap_path: str, scale: float) -> Path:
    """キャッシュ画像の保存先を決める.

    dataset_id で分けているのは、同じファイル名の basemap を持つ
    別データセットが混ざらないようにするため。
    scale をファイル名に含めているのは、倍率を変えたときに
    古いキャッシュを読んでしまわないようにするため。
    """
    settings = get_settings()
    stem = Path(basemap_path).stem
    suffix = Path(basemap_path).suffix or ".png"
    return (
        settings.DERIVED_ROOT / "basemap_cache" / dataset_id
        / f"{stem}_x{scale:g}{suffix}"
    )


def load_basemap(
    dataset_id: str,
    dataroot: str,
    basemap_path: str,
    scale: float = DEFAULT_SCALE,
) -> Image.Image | None:
    """basemap を縮小して返す（キャッシュがあればそれを使う）.

    Args:
        dataset_id: Dataset.id
        dataroot:   Dataset.dataroot（DATA_ROOT からの相対パス）
        basemap_path: MapMeta.basemap_path（dataroot からの相対パス）
        scale: 縮小倍率

    Returns:
        縮小済み画像。元画像が見つからない場合は None
        （Map Expansion 無しの構成でも画面が落ちないようにするため）
    """
    settings = get_settings()
    cached = cache_path(dataset_id, basemap_path, scale)
    if cached.exists():
        return Image.open(cached)

    source = settings.DATA_ROOT / dataroot / basemap_path
    if not source.exists():
        logger.warning("basemap not found: %s", source)
        return None

    logger.info("building basemap cache: %s -> %s", source, cached)
    cached.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        w, h = img.size
        resized = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
        # RGBA のままだと PNG 保存で巨大になることがあるので落とす
        if resized.mode not in ("RGB", "L"):
            resized = resized.convert("RGB")
        resized.save(cached)
    return resized


def clear_cache(dataset_id: str | None = None) -> int:
    """キャッシュを削除する。削除したファイル数を返す."""
    settings = get_settings()
    base = settings.DERIVED_ROOT / "basemap_cache"
    target = base / dataset_id if dataset_id else base
    if not target.exists():
        return 0
    count = 0
    for p in target.rglob("*"):
        if p.is_file():
            p.unlink()
            count += 1
    return count
