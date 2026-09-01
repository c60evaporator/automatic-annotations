"""ラベル体系の変換.

config には「ラベル → グループ」「ラベル → nuScenes カテゴリ」の
向きで持たせてある。1ラベルに対する行き先が1つに定まるので、
設定として書きやすく、矛盾も起きにくいため。

一方で推論リクエストには「グループ → ラベル一覧」が要る
（グループ単位でプロンプトをまとめて投げるため）。
その逆引きと、設定の整合チェックをここに集約する。
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from app.core.config import get_settings


@lru_cache(maxsize=1)
def label_groups() -> dict[str, list[str]]:
    """グループ → ラベル一覧（LABEL_TO_CATEGORY_GROUP の逆引き）.

    グループの並びは config の登場順、ラベルは辞書順にして、
    実行のたびにプロンプトの順序が変わらないようにする
    （順序が揺れると検出結果の再現性が落ちる）。
    """
    settings = get_settings()
    groups: dict[str, list[str]] = defaultdict(list)
    for label, group in settings.LABEL_TO_CATEGORY_GROUP.items():
        groups[group].append(label)
    return {g: sorted(labels) for g, labels in groups.items()}


@lru_cache(maxsize=1)
def group_names() -> list[str]:
    return list(label_groups().keys())


@lru_cache(maxsize=1)
def all_labels() -> list[str]:
    """全ラベルをグループ順・ラベル順に並べて返す.

    凡例の並びに使う。検出結果に現れたラベルだけを出すと、
    推論のたびに並びとチェック状態が変わってしまうため、
    設定側の定義を正とする。
    """
    return [label for labels in label_groups().values() for label in labels]


def group_of(label: str) -> str | None:
    """ラベルが属するカテゴリグループ."""
    return get_settings().LABEL_TO_CATEGORY_GROUP.get(label)


def label_to_nusc_category(label: str) -> str | None:
    """検出ラベル → nuScenes の category_name.

    3D ボックスを SampleAnnotation として保存するとき、
    Category テーブルの name に合わせるために使う。
    """
    return get_settings().LABEL_TO_NUSC_CATEGORY.get(label)


def nusc_category_to_label(category_name: str) -> str | None:
    """nuScenes の category_name → 検出ラベル.

    GT と自動アノテーションを比較するときに、
    双方を同じラベル空間へ寄せるために使う。
    """
    return get_settings().NUSC_CATEGORY_TO_LABEL.get(category_name)


def scaled_score_thresholds(ratio: float = 1.0) -> dict[str, float]:
    """グループごとのスコア閾値（UI の Ratio スライダーを掛けた値）."""
    return {
        g: round(v * ratio, 3)
        for g, v in get_settings().DET2D_DEFAULT_SCORE_THRESHOLDS.items()
    }


def scaled_nms_same_class_ious(ratio: float = 1.0) -> dict[str, float]:
    """グループごとの同一クラス NMS IoU."""
    return {
        g: round(min(v * ratio, 1.0), 3)
        for g, v in get_settings().DET2D_NMS_SAME_CLASS_IOUS.items()
    }


def validate_label_config() -> list[str]:
    """設定の取りこぼしを洗い出す.

    ラベルを1つ足したときに、閾値やカテゴリ変換の追加を忘れると、
    そのラベルだけ既定値で走ったり、DB 保存時に落ちたりする。
    UI 側で警告を出せるよう、問題を文字列のリストで返す。
    """
    settings = get_settings()
    problems: list[str] = []

    groups = label_groups()

    # グループに閾値が定義されているか
    for group in groups:
        if group not in settings.DET2D_DEFAULT_SCORE_THRESHOLDS:
            problems.append(f"グループ '{group}' のスコア閾値が未定義です")
        if group not in settings.DET2D_NMS_SAME_CLASS_IOUS:
            problems.append(f"グループ '{group}' の NMS IoU が未定義です")

    # 閾値だけあってグループが存在しない（綴り違い等）
    for group in settings.DET2D_DEFAULT_SCORE_THRESHOLDS:
        if group not in groups:
            problems.append(
                f"スコア閾値の '{group}' に対応するカテゴリグループがありません"
            )

    # 検出ラベルが nuScenes カテゴリへ変換できるか
    for label in settings.LABEL_TO_CATEGORY_GROUP:
        if label not in settings.LABEL_TO_NUSC_CATEGORY:
            problems.append(
                f"ラベル '{label}' の nuScenes カテゴリ変換が未定義です"
            )

    # nuScenes カテゴリからの変換先が検出ラベルとして存在するか
    for category, label in settings.NUSC_CATEGORY_TO_LABEL.items():
        if label not in settings.LABEL_TO_CATEGORY_GROUP:
            problems.append(
                f"カテゴリ '{category}' の変換先ラベル '{label}' が"
                "カテゴリグループに登録されていません"
            )

    return problems


def clear_caches() -> None:
    """設定を差し替えたときに呼ぶ（テスト用）."""
    label_groups.cache_clear()
    group_names.cache_clear()
    all_labels.cache_clear()
