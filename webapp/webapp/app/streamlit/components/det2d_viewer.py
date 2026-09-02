"""カメラ画像上に 2D bounding box を描画するコンポーネント.

ラベルは画像に文字で焼き込まず、色と凡例で区別する。
1600x900 を2列で並べると、焼き込んだ文字が潰れて読めないため。
"""
from __future__ import annotations

import hashlib
import html
from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence

import streamlit as st
from PIL import Image, ImageDraw

# 彩度・明度を揃えた10色。隣り合っても見分けがつくよう色相を散らしてある
PALETTE = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#00b8c4",  # cyan
    "#f032e6",  # magenta
    "#9a9a06",  # olive
    "#008080",  # teal
    "#a05a2c",  # brown
]

# ── 凡例の見た目の調整値 ──────────────────────────────────────────────────────
# 詰まり具合と縦位置はここの数値だけで調整できる。単位は rem（SWATCH のみ px）。
#
#   LEGEND_ROW_GAP    : ラベル行どうしの間隔。
#                       0 にすると行が接触し、見出しとボタンも重なる
#   LEGEND_HEADER_GAP : 「Labels」見出しと「全て/解除」ボタンの間隔
#   LEGEND_BUTTON_GAP : ボタン行とラベル1行目の間隔
#   LEGEND_ROW_HEIGHT : 1行の高さ。行内の全要素にこの高さを与えて
#                       中央寄せするので、チェックボックスと文字がずれない。
#                       下げすぎるとチェックボックスが欠ける
#   LEGEND_CHIP_FONT  : ラベル文字の大きさ
#   LEGEND_CHIP_NUDGE : それでも1〜2px ずれる場合の微調整。
#                       正の値でチップが下、負の値で上に動く
#   LEGEND_CHECK_GAP  : チェックボックスと色チップの横間隔。
#                       "xxsmall"〜"xxlarge" の名前か、ピクセル数の整数を渡す
LEGEND_ROW_GAP = 0.2
LEGEND_HEADER_GAP = 1.8
LEGEND_BUTTON_GAP = 0.8
LEGEND_ROW_HEIGHT = 1.0
LEGEND_CHIP_FONT = 0.7
LEGEND_CHIP_NUDGE = -0.4
LEGEND_CHECK_GAP = "xsmall"
LEGEND_SWATCH_PX = 12

# ── ボックス描画の調整値 ──────────────────────────────────────────────────────
#   BOX_LINE_WIDTH      : 枠線の太さ（px）
#   BOX_TEXT_SIZE       : 画像に重ねる文字の大きさ（px）。
#                         1600x900 のカメラ画像を想定した値
#   BOX_TEXT_STROKE     : 文字の縁取り（px）。0 で無効。1 が既定。
#                         文字色を枠線と同色にすると背景に溶けるので、
#                         暗い縁を付けて可読性を確保する
#   BOX_TEXT_MARGIN     : ボックス上端と文字の間隔（px）
BOX_LINE_WIDTH = 5
BOX_TEXT_SIZE = 26
BOX_TEXT_STROKE = 1
BOX_TEXT_STROKE_COLOR = "#000000"
BOX_TEXT_MARGIN = 2

# 画像に重ねる文字の種類
TEXT_MODE_NONE = "None"
TEXT_MODE_LABEL = "Label"
TEXT_MODE_SCORE = "Score"
TEXT_MODES = (TEXT_MODE_NONE, TEXT_MODE_LABEL, TEXT_MODE_SCORE)


@lru_cache(maxsize=1)
def _label_index() -> dict[str, int]:
    """設定上のラベル定義順 → インデックス.

    色はこの順に配る。ハッシュで決めると、ラベル数がパレット数以下でも
    衝突して同じ色が複数ラベルに割り当たる（10ラベル10色でも実際に
    4ラベルが同色になった）。定義順なら数が収まる限り必ず重複しない。
    """
    from app.services.label_service import all_labels

    return {label: i for i, label in enumerate(all_labels())}


def color_for(label: str) -> str:
    """ラベルから色を決める.

    設定に無いラベル（推論が想定外の語を返した場合など）は md5 で決める。

    NOTE: 組み込みの hash() は使わないこと。
    文字列の hash はプロセスごとにシードが変わるため、
    サーバーを再起動するたびに色が入れ替わってしまう。
    """
    index = _label_index().get(label)
    if index is None:
        index = int(hashlib.md5(label.encode("utf-8")).hexdigest()[:8], 16)
    return PALETTE[index % len(PALETTE)]


def clear_color_cache() -> None:
    """ラベル定義を変更したときに呼ぶ（テスト用）."""
    _label_index.cache_clear()


def passes_score(box: dict[str, Any], min_score: float) -> bool:
    """スコア閾値を満たすか.

    Ground truth のボックスは score を持たない（None）。
    None を 0 とみなすと閾値を上げた瞬間に GT が全部消えてしまうため、
    スコアが無いボックスは閾値の対象外として常に通す。
    """
    score = box.get("score")
    return score is None or score >= min_score


def filter_boxes(
    boxes: Iterable[dict[str, Any]],
    *,
    min_score: float = 0.0,
    enabled_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    """スコアと表示ラベルでボックスを絞る."""
    out = []
    for box in boxes:
        if not passes_score(box, min_score):
            continue
        if enabled_labels is not None and box["label"] not in enabled_labels:
            continue
        out.append(box)
    return out


@lru_cache(maxsize=4)
def _font(size: int):
    """描画用フォント.

    load_default(size=...) は Pillow 10.1 以降でスケーラブルなフォントを返す。
    システムに特定の TTF があることを前提にしないので、
    コンテナのベースイメージが変わっても動く。
    """
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # 古い Pillow ではサイズ指定ができない（ビットマップフォントになる）
        return ImageFont.load_default()


def box_text(box: dict[str, Any], text_mode: str) -> str:
    """ボックスに重ねる文字を作る."""
    if text_mode == TEXT_MODE_LABEL:
        return str(box.get("label", ""))
    if text_mode == TEXT_MODE_SCORE:
        score = box.get("score")
        return "" if score is None else f"{score:.2f}"
    return ""


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[dict[str, Any]],
    *,
    width: int | None = None,
    text_mode: str = TEXT_MODE_NONE,
) -> Image.Image:
    """画像に BBox を描いた新しい画像を返す.

    Args:
        width: 枠線の太さ。None なら BOX_LINE_WIDTH
        text_mode: TEXT_MODE_NONE / _LABEL / _SCORE。
            文字色は枠線と同じ色にする（凡例の色と対応させるため）

    元画像はキャッシュ（cache_resource）で共有されているので、
    必ず copy してから描く。直接描くとキャッシュ内の画像が汚染され、
    再描画のたびにボックスが重なって増えていく。
    """
    line_width = BOX_LINE_WIDTH if width is None else width
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(BOX_TEXT_SIZE) if text_mode != TEXT_MODE_NONE else None

    for box in boxes:
        color = color_for(box["label"])
        draw.rectangle(
            (box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
            outline=color,
            width=line_width,
        )

        if font is None:
            continue
        text = box_text(box, text_mode)
        if not text:
            continue

        # ボックスの上に置く。上端に収まらない場合は内側へ回り込ませる
        _, top, _, bottom = draw.textbbox((0, 0), text, font=font)
        text_h = bottom - top
        y = box["ymin"] - text_h - BOX_TEXT_MARGIN
        if y < 0:
            y = box["ymin"] + BOX_TEXT_MARGIN
        draw.text(
            (box["xmin"], y),
            text,
            font=font,
            fill=color,
            stroke_width=BOX_TEXT_STROKE,
            stroke_fill=BOX_TEXT_STROKE_COLOR,
        )

    return canvas


# ── 凡例 ──────────────────────────────────────────────────────────────────────

def _chip(
    label: str,
    count: int | None = None,
    color_fn: Callable[[str], str] = color_for,
) -> str:
    """色見本つきのラベルチップ（HTML）.

    高さをチェックボックスと揃えることで、縦位置のずれを防ぐ。
    """
    text = html.escape(label)
    if count is not None:
        text += f" ({count})"
    # 高さは親（行）に合わせて 100%。固定値にすると、行の高さを変えたときに
    # チェックボックス側とずれる
    nudge = f"position:relative;top:{LEGEND_CHIP_NUDGE}rem;" if LEGEND_CHIP_NUDGE else ""
    return (
        f'<div style="display:flex;align-items:center;gap:6px;'
        f'height:100%;line-height:1;{nudge}">'
        f'<span style="width:{LEGEND_SWATCH_PX}px;height:{LEGEND_SWATCH_PX}px;'
        f"border-radius:3px;flex:0 0 {LEGEND_SWATCH_PX}px;"
        f'background:{color_for(label)};"></span>'
        f'<span style="font-size:{LEGEND_CHIP_FONT}rem;white-space:nowrap;">{text}</span>'
        "</div>"
    )


def _disabled_key(key_prefix: str) -> str:
    return f"{key_prefix}__disabled"


def _set_all(labels: Sequence[str], key_prefix: str, value: bool) -> None:
    """凡例のチェックを一括で切り替える（on_click コールバック）.

    コールバックはスクリプト再実行の前に走るので、
    ウィジェットの key に直接書き込んでよい。
    描画後に書き込むと StreamlitAPIException になる。
    """
    disabled: set[str] = st.session_state.setdefault(_disabled_key(key_prefix), set())
    for label in labels:
        st.session_state[f"{key_prefix}_{label}"] = value
        if value:
            disabled.discard(label)
        else:
            disabled.add(label)


def _legend_css(key: str) -> str:
    """凡例だけに効く CSS.

    st.container(key=...) を使うと要素に .st-key-<key> クラスが付くので、
    ページ全体ではなく凡例だけにスタイルを当てられる。
    グローバルに書くと他のチェックボックスやボタンまで詰まってしまう。

    縦位置を揃える考え方:
    チェックボックスと色チップは、内部構造も自然な高さも異なる。
    片方だけに高さを指定したり vertical_alignment に頼ったりすると必ずずれる。
    **行内の全要素コンテナに同じ高さを与え、その中で中央寄せする**のが確実。

    NOTE: margin を一律 0 にしてはいけない。
    見出しとボタンが重なり、行も接触する。
    """
    return f"""
<style>
/* --- 行（横並びコンテナ）そのもの --- */
div.st-key-{key} [data-testid="stHorizontalBlock"] {{
    align-items: center !important;
    margin-bottom: {LEGEND_ROW_GAP}rem !important;
}}
/* ボタン行の下だけ少し広く空ける */
div.st-key-{key} [data-testid="stHorizontalBlock"]:has([data-testid="stButton"]) {{
    margin-bottom: {LEGEND_BUTTON_GAP}rem !important;
}}

/* --- 行内の要素をすべて同じ高さの中央寄せにする（縦位置ずれの対策） --- */
div.st-key-{key} [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"] {{
    display: flex !important;
    align-items: center !important;
    height: {LEGEND_ROW_HEIGHT}rem !important;
    margin: 0 !important;
}}
div.st-key-{key} [data-testid="stCheckbox"] {{
    display: flex !important;
    align-items: center !important;
    height: 100% !important;
    min-height: 0 !important;
}}
div.st-key-{key} [data-testid="stCheckbox"] label {{
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
}}
div.st-key-{key} [data-testid="stMarkdownContainer"] {{
    display: flex !important;
    align-items: center !important;
    height: 100% !important;
}}
div.st-key-{key} [data-testid="stMarkdownContainer"] p {{
    margin: 0 !important;
}}

/* --- 見出し --- */
div.st-key-{key} .legend-header {{
    font-weight: 600;
    line-height: 1.2;
    margin: 0 0 {LEGEND_HEADER_GAP}rem 0;
}}

/* --- ボタン --- */
div.st-key-{key} [data-testid="stButton"] button {{
    padding: 0.1rem 0.5rem !important;
    min-height: 0 !important;
    line-height: 1.4 !important;
    font-size: 0.75rem !important;
}}
</style>
"""


def render_label_legend(
    labels: Sequence[str],
    *,
    counts: dict[str, int] | None = None,
    key_prefix: str = "det2d_legend",
    show_toggle_all: bool = True,
    color_fn: Callable[[str], str] = color_for,
    header: str = "Labels",
) -> set[str]:
    """色見本つきのチェックボックス凡例を描画し、有効な項目を返す.

    color_fn を差し替えることで、ラベル色分け以外（Track ID 色分けなど）
    にも使い回せる。

    ラベルの一覧は config の定義から渡す想定。検出結果に含まれるものだけを
    出すと、推論のたびに凡例の並びとチェック状態が変わってしまう。

    Args:
        labels: 表示するラベル
        counts: 現在表示中のボックス数。ラベルの横に出す

    Returns:
        チェックが入っているラベルの集合
    """
    st.markdown(_legend_css(key_prefix), unsafe_allow_html=True)

    # チェックを外したラベルを、ウィジェットとは別のキーに覚えておく。
    # 表示するラベルは sample ごとに変わるため、描画されなかった
    # チェックボックスの状態は Streamlit に破棄される。
    # ウィジェットの key だけに頼ると、sample を切り替えて戻ったときに
    # 「外したはずのラベル」が復活してしまう
    disabled: set[str] = st.session_state.setdefault(_disabled_key(key_prefix), set())

    enabled: set[str] = set()

    # key を付けると .st-key-<key> で CSS を絞れる。
    # 縦の間隔は CSS 側（LEGEND_ROW_GAP）で制御するので gap は 0 にする
    with st.container(key=key_prefix, gap=None):
        st.markdown(f'<div class="legend-header">{html.escape(header)}</div>',
                    unsafe_allow_html=True)

        if show_toggle_all:
            # 横並びコンテナ + width="content" で、ボタンをラベル幅だけにする。
            # st.columns だと列幅が等分され、狭い列では折り返してしまう
            with st.container(horizontal=True, gap="small"):
                st.button(
                    "全て", key=f"{key_prefix}_all", width="content",
                    on_click=_set_all, args=(labels, key_prefix, True),
                )
                st.button(
                    "解除", key=f"{key_prefix}_none", width="content",
                    on_click=_set_all, args=(labels, key_prefix, False),
                )

        for label in labels:
            key = f"{key_prefix}_{label}"
            # 既定値は value= ではなく session_state で与える。
            # value= と「一括切り替えボタンによる session_state への書き込み」を
            # 併用すると、Streamlit が
            # "created with a default value but also had its value set via
            #  the Session State API" と警告を出す
            if key not in st.session_state:
                st.session_state[key] = label not in disabled

            with st.container(
                horizontal=True,
                vertical_alignment="center",
                gap=LEGEND_CHECK_GAP,
            ):
                checked = st.checkbox(
                    label, key=key, label_visibility="collapsed", width="content"
                )
                st.markdown(
                    _chip(label, None if counts is None else counts.get(label, 0)),
                    unsafe_allow_html=True,
                )
            if checked:
                enabled.add(label)
                disabled.discard(label)
            else:
                disabled.add(label)
    return enabled


def labels_in_items(
    items: Iterable[dict[str, Any]], *, min_score: float = 0.0
) -> list[str]:
    """items に含まれるラベルを、設定の定義順で返す.

    凡例をこの結果に絞ると、いま見ているサンプルに出ているものだけが並ぶ。
    設定順に固定するのは、サンプルを切り替えるたびに並びが入れ替わると
    追いにくいため（検出順や出現順にはしない）。
    """
    found = {
        box["label"]
        for item in items
        for box in (item.get("boxes") or [])
        if passes_score(box, min_score)
    }
    order = _label_index()
    known = sorted((l for l in found if l in order), key=lambda l: order[l])
    # 設定に無いラベル（推論が想定外の語を返した場合）は末尾に回す
    unknown = sorted(l for l in found if l not in order)
    return known + unknown


def count_by_label(
    items: Iterable[dict[str, Any]], *, min_score: float = 0.0
) -> dict[str, int]:
    """カメラグリッドの items からラベルごとのボックス数を数える.

    凡例に出す件数用。ラベルのチェック状態は数に反映しない
    （チェックを外した瞬間に件数が 0 になると、戻す判断ができなくなる）。
    """
    counts: dict[str, int] = {}
    for item in items:
        for box in item.get("boxes") or []:
            if not passes_score(box, min_score):
                continue
            counts[box["label"]] = counts.get(box["label"], 0) + 1
    return counts


def render_camera_grid(
    items: list[dict[str, Any]],
    *,
    columns: int = 2,
    min_score: float = 0.0,
    enabled_labels: set[str] | None = None,
    show_boxes: bool = True,
    text_mode: str = TEXT_MODE_NONE,
) -> None:
    """カメラ画像をグリッド表示する.

    items の各要素:
        {"channel": str, "image": PIL.Image | None,
         "boxes": [...], "pending": bool}
    """
    cols = st.columns(columns)
    for i, item in enumerate(items):
        with cols[i % columns]:
            _render_one(
                item, item.get("boxes"), item["channel"],
                min_score=min_score, enabled_labels=enabled_labels,
                show_boxes=show_boxes, text_mode=text_mode,
                pending=item.get("pending", False),
            )


def render_camera_comparison_grid(
    items: list[dict[str, Any]],
    *,
    min_score: float = 0.0,
    enabled_labels: set[str] | None = None,
    show_boxes: bool = True,
    text_mode: str = TEXT_MODE_NONE,
    left_label: str = "GT",
    right_label: str = "Pred",
) -> None:
    """1カメラにつき1行、左に Ground truth・右に推論結果を並べる.

    items の各要素は render_camera_grid と同じ形に加えて "gt_boxes" を持つ。
    左右で同じ画像を使うので、読み込みは1回で済ませている
    （描画時に copy されるため、片方のボックスがもう片方に混ざることはない）。
    """
    for item in items:
        left_col, right_col = st.columns(2)
        with left_col:
            _render_one(
                item, item.get("gt_boxes"), f"{item['channel']} [{left_label}]",
                min_score=0.0,  # GT にスコアは無いので閾値を適用しない
                enabled_labels=enabled_labels,
                show_boxes=show_boxes, text_mode=text_mode, pending=False,
            )
        with right_col:
            _render_one(
                item, item.get("boxes"), f"{item['channel']} [{right_label}]",
                min_score=min_score, enabled_labels=enabled_labels,
                show_boxes=show_boxes, text_mode=text_mode,
                pending=item.get("pending", False),
            )


def _render_one(
    item: dict[str, Any],
    boxes: Sequence[dict[str, Any]] | None,
    caption: str,
    *,
    min_score: float,
    enabled_labels: set[str] | None,
    show_boxes: bool,
    text_mode: str,
    pending: bool,
) -> None:
    """画像1枚を描画する（グリッド共通の処理）."""
    image = item.get("image")
    if image is None:
        st.warning(f"{item['channel']}: 画像が見つかりません")
        return

    if show_boxes:
        shown = filter_boxes(
            boxes or [], min_score=min_score, enabled_labels=enabled_labels
        )
        caption += "（推論待ち）" if pending else f"  {len(shown)} boxes"
        if shown:
            image = draw_boxes(image, shown, text_mode=text_mode)
    elif pending:
        caption += "（推論待ち）"

    st.image(image, caption=caption, width="stretch")
