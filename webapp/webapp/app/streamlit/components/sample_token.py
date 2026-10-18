"""選択中の sample_token を、表示オプション列の一番上に 1 行で出す.

``sample=<token>`` の形で出す（tools/check_depth_boxfitting.py の --sample に
そのまま渡せる値）。

調査時に DB やスクリプトと突き合わせるためのもの。列が狭いので
はみ出した分は「…」で省略し、高さを取らないようにしている。
全文はマウスを乗せると表示され（title 属性）、トリプルクリックで
全文を選択してコピーできる（省略は表示上だけで、テキストは全文ある）。
"""
from __future__ import annotations

import html

import streamlit as st


def render_sample_token(token: str | None) -> None:
    """sample_token を 1 行で表示する."""
    if not token:
        return
    safe = html.escape(str(token))
    st.markdown(
        f'<div title="sample_token: {safe}" style="white-space:nowrap;'
        'overflow:hidden;text-overflow:ellipsis;font-size:0.75rem;'
        'line-height:1.2;color:rgba(128,128,128,0.95);margin-bottom:0.25rem;">'
        f'<code style="font-size:0.75rem;padding:0;background:none;">sample={safe}</code>'
        '</div>',
        unsafe_allow_html=True,
    )
