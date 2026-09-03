"""カメラ画像上にインスタンスマスクとトラックを描画するコンポーネント.

Detection2D の描画（枠のみ）と違い、マスクを半透明で重ねる。
色分けは「ラベル」と「Track ID」を切り替えられる。
"""
from __future__ import annotations

import colorsys
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from app.streamlit.components.det2d_viewer import (
    BOX_LINE_WIDTH,
    BOX_TEXT_MARGIN,
    BOX_TEXT_SIZE,
    BOX_TEXT_STROKE,
    BOX_TEXT_STROKE_COLOR,
    _font,
)
from app.streamlit.components.det2d_viewer import color_for as color_for_label
from common.mask_rle import decode_rle

# ── 表示モード ────────────────────────────────────────────────────────────────

# 画像に重ねるボックスの種類
BOX_MODE_PROMPT = "Prompt"      # プロンプトに使った Detection2D のボックス
BOX_MODE_INSTANCE = "Instance"  # マスクの外接矩形
BOX_MODE_NONE = "None"
BOX_MODES = (BOX_MODE_PROMPT, BOX_MODE_INSTANCE, BOX_MODE_NONE)

# 色の決め方
COLOR_MODE_LABEL = "Label"
COLOR_MODE_TRACK = "Track ID"
COLOR_MODES = (COLOR_MODE_LABEL, COLOR_MODE_TRACK)

# インスタンスの上に出す文字
TEXT_MODE_NONE = "None"
TEXT_MODE_LABEL = "Label"
TEXT_MODE_TRACK = "Track ID"
INSTANCE_TEXT_MODES = (TEXT_MODE_NONE, TEXT_MODE_LABEL, TEXT_MODE_TRACK)

# マスクの重ね合わせの濃さ
MASK_ALPHA = 0.45

# インスタンスの由来（DB の models と同じ値）
ORIGIN_PROMPT = "prompt"
ORIGIN_PROPAGATED = "propagated"


def split_by_origin(
    instances: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """インスタンスを (prompt 由来, propagated 由来) に分ける."""
    prompt: list[dict[str, Any]] = []
    propagated: list[dict[str, Any]] = []
    for inst in instances:
        if inst.get("origin") == ORIGIN_PROPAGATED:
            propagated.append(inst)
        else:
            prompt.append(inst)
    return prompt, propagated


def preferred_instances(
    instances: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """通常表示に使うインスタンスを選ぶ.

    区間境界の sample には prompt と propagated の両方があるので、
    検出器の出力そのものである prompt を優先する。
    それ以外の sample は propagated しかないのでそのまま返る。
    """
    prompt, propagated = split_by_origin(instances)
    return prompt if prompt else propagated


# ── 色 ────────────────────────────────────────────────────────────────────────

def color_for_track(track_id: str) -> str:
    """Track ID から色を決める.

    track_id は数百まで増えうるので、固定パレットの使い回しでは
    隣接トラックが同色になりやすい。黄金比で色相を回して、
    連番でも隣り合う色が離れるようにする。

    NOTE: 組み込みの hash() は使わない。プロセスごとにシードが変わり、
    サーバー再起動で色が入れ替わってしまう。
    """
    try:
        index = int(track_id)
    except (TypeError, ValueError):
        index = int(hashlib.md5(str(track_id).encode()).hexdigest()[:8], 16)

    # 0.618... （黄金比の小数部）ずつ回すと、連番でも色相が大きく離れる
    hue = (index * 0.61803398875) % 1.0
    # 彩度・明度は固定して、同じ見え方の濃さに揃える
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def color_for_instance(instance: dict[str, Any], color_mode: str) -> str:
    if color_mode == COLOR_MODE_TRACK:
        return color_for_track(instance.get("track_id", ""))
    return color_for_label(instance.get("label", ""))


def legend_key(instance: dict[str, Any], color_mode: str) -> str:
    """凡例・フィルタで使うキー（色分けの単位）."""
    if color_mode == COLOR_MODE_TRACK:
        return str(instance.get("track_id", ""))
    return str(instance.get("label", ""))


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


# ── フィルタ・集計 ────────────────────────────────────────────────────────────

def filter_instances(
    instances: Iterable[dict[str, Any]],
    *,
    color_mode: str,
    enabled_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    if enabled_keys is None:
        return list(instances)
    return [
        inst for inst in instances
        if legend_key(inst, color_mode) in enabled_keys
    ]


def legend_entries(
    items: Iterable[dict[str, Any]], color_mode: str
) -> tuple[list[str], dict[str, int]]:
    """表示中のインスタンスから凡例の項目と件数を作る.

    Returns:
        (キー一覧, キーごとの件数)。
        Track ID は数値順、ラベルは設定の定義順で並べる。
    """
    counts: dict[str, int] = {}
    for item in items:
        for inst in item.get("instances") or []:
            key = legend_key(inst, color_mode)
            counts[key] = counts.get(key, 0) + 1

    if color_mode == COLOR_MODE_TRACK:
        # 文字列のままだと "10" < "2" になるので数値として並べる
        def sort_key(k: str):
            try:
                return (0, int(k))
            except ValueError:
                return (1, k)
        keys = sorted(counts, key=sort_key)
    else:
        from app.streamlit.components.det2d_viewer import _label_index
        order = _label_index()
        known = sorted((k for k in counts if k in order), key=lambda k: order[k])
        keys = known + sorted(k for k in counts if k not in order)
    return keys, counts


# ── 描画 ──────────────────────────────────────────────────────────────────────

def draw_instances(
    image: Image.Image,
    instances: Sequence[dict[str, Any]],
    *,
    color_mode: str = COLOR_MODE_LABEL,
    box_mode: str = BOX_MODE_INSTANCE,
    text_mode: str = TEXT_MODE_TRACK,
    prompt_boxes: Sequence[dict[str, Any]] | None = None,
    mask_alpha: float = MASK_ALPHA,
    line_width: int | None = None,
) -> Image.Image:
    """マスクとボックスを重ねた新しい画像を返す.

    Args:
        prompt_boxes: box_mode='Prompt' のときに描く Detection2D のボックス
        mask_alpha: マスクの濃さ（0 で非表示）

    元画像はキャッシュで共有されているので、必ずコピーしてから描く。
    """
    width_px = line_width if line_width is not None else BOX_LINE_WIDTH

    # --- マスクの重ね合わせ ---
    if mask_alpha > 0 and instances:
        canvas_array = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = canvas_array.shape[:2]
        drew = False
        for inst in instances:
            rle = inst.get("mask_rle")
            if not rle:
                continue
            if tuple(rle["size"]) != (height, width):
                # 画像とマスクの解像度が食い違う場合は重ねない。
                # 引き伸ばすと位置がずれて誤解を生む
                continue
            mask = decode_rle(rle)
            color = np.array(_hex_to_rgb(color_for_instance(inst, color_mode)),
                             dtype=np.float32)
            canvas_array[mask] = (
                canvas_array[mask] * (1.0 - mask_alpha) + color * mask_alpha
            )
            drew = True
        canvas = (
            Image.fromarray(canvas_array.astype(np.uint8)) if drew else image.copy()
        )
    else:
        canvas = image.copy()

    draw = ImageDraw.Draw(canvas)
    font = _font(BOX_TEXT_SIZE) if text_mode != TEXT_MODE_NONE else None

    # --- ボックス ---
    if box_mode == BOX_MODE_PROMPT:
        for box in prompt_boxes or []:
            color = color_for_label(box.get("label", ""))
            draw.rectangle(
                (box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
                outline=color, width=width_px,
            )
    elif box_mode == BOX_MODE_INSTANCE:
        for inst in instances:
            draw.rectangle(
                (inst["xmin"], inst["ymin"], inst["xmax"], inst["ymax"]),
                outline=color_for_instance(inst, color_mode), width=width_px,
            )

    # --- 文字 ---
    if font is not None:
        for inst in instances:
            text = (
                str(inst.get("label", "")) if text_mode == TEXT_MODE_LABEL
                else str(inst.get("track_id", ""))
            )
            if not text:
                continue
            color = color_for_instance(inst, color_mode)
            _, top, _, bottom = draw.textbbox((0, 0), text, font=font)
            text_height = bottom - top
            y = inst["ymin"] - text_height - BOX_TEXT_MARGIN
            if y < 0:
                y = inst["ymin"] + BOX_TEXT_MARGIN
            draw.text(
                (inst["xmin"], y), text, font=font, fill=color,
                stroke_width=BOX_TEXT_STROKE, stroke_fill=BOX_TEXT_STROKE_COLOR,
            )

    return canvas


def render_instance_comparison_grid(
    items: list[dict[str, Any]],
    *,
    color_mode: str = COLOR_MODE_LABEL,
    box_mode: str = BOX_MODE_INSTANCE,
    text_mode: str = TEXT_MODE_TRACK,
    enabled_keys: set[str] | None = None,
    mask_alpha: float = MASK_ALPHA,
    left_label: str = "Propagated",
    right_label: str = "Prompt",
) -> None:
    """1 カメラにつき 1 行、左に伝播マスク・右に推論マスクを並べる.

    items の各要素は render_instance_grid と同じ形に加えて
    "propagated_instances" / "prompt_instances" を持つ。

    左（伝播）にはプロンプトボックスを描かない。
    プロンプトは右側の入力であり、左に重ねると
    「伝播がどれだけずれたか」が読み取れなくなる。
    """
    for item in items:
        left_col, right_col = st.columns(2)
        with left_col:
            _render_one(
                item, item.get("propagated_instances") or [],
                f"{item['channel']} [{left_label}]",
                color_mode=color_mode,
                # 伝播側にプロンプト枠は出さない
                box_mode=(BOX_MODE_NONE if box_mode == BOX_MODE_PROMPT else box_mode),
                text_mode=text_mode, enabled_keys=enabled_keys,
                prompt_boxes=None, mask_alpha=mask_alpha, pending=False,
            )
        with right_col:
            _render_one(
                item, item.get("prompt_instances") or [],
                f"{item['channel']} [{right_label}]",
                color_mode=color_mode, box_mode=box_mode,
                text_mode=text_mode, enabled_keys=enabled_keys,
                prompt_boxes=item.get("prompt_boxes"),
                mask_alpha=mask_alpha, pending=item.get("pending", False),
            )


def _render_one(
    item: dict[str, Any],
    instances: Sequence[dict[str, Any]],
    caption: str,
    *,
    color_mode: str,
    box_mode: str,
    text_mode: str,
    enabled_keys: set[str] | None,
    prompt_boxes: Sequence[dict[str, Any]] | None,
    mask_alpha: float,
    pending: bool,
) -> None:
    """画像 1 枚を描画する（グリッド共通）."""
    image = item.get("image")
    if image is None:
        st.warning(f"{item['channel']}: 画像が見つかりません")
        return

    shown = filter_instances(
        instances, color_mode=color_mode, enabled_keys=enabled_keys
    )
    caption += "（推論待ち）" if pending else f"  {len(shown)} instances"

    rendered = draw_instances(
        image, shown,
        color_mode=color_mode, box_mode=box_mode, text_mode=text_mode,
        prompt_boxes=prompt_boxes, mask_alpha=mask_alpha,
    )
    st.image(rendered, caption=caption, width="stretch")


def render_instance_grid(
    items: list[dict[str, Any]],
    *,
    columns: int = 2,
    color_mode: str = COLOR_MODE_LABEL,
    box_mode: str = BOX_MODE_INSTANCE,
    text_mode: str = TEXT_MODE_TRACK,
    enabled_keys: set[str] | None = None,
    mask_alpha: float = MASK_ALPHA,
) -> None:
    """カメラ画像をグリッド表示する.

    items の各要素:
        {"channel": str, "image": PIL.Image | None,
         "instances": [...], "prompt_boxes": [...], "pending": bool}
    """
    cols = st.columns(columns)
    for i, item in enumerate(items):
        with cols[i % columns]:
            _render_one(
                item, item.get("instances") or [], item["channel"],
                color_mode=color_mode, box_mode=box_mode, text_mode=text_mode,
                enabled_keys=enabled_keys,
                prompt_boxes=item.get("prompt_boxes"),
                mask_alpha=mask_alpha, pending=item.get("pending", False),
            )
