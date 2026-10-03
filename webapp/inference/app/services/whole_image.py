"""検出が 0 件のフレームを、画像全体の分類で救済する.

## なぜ必要か

トラックの荷台が視野いっぱいに写るような構図では、GroundingDINO が
何も検出しないことがある。物体の輪郭が画面内に収まっていないため、
「物体らしさ」を捉えられない。

そこで **検出が 0 件のカメラ・フレームに限って**、画像全体を 1 枚の
切り出しとみなして SigLIP2 で分類する。

## 「どのラベル候補でもない」の扱い

2 段で落とす。

1. **除外候補を混ぜる**（road / building など）。これが最大なら検出なし。
   相対比較なので閾値の較正に依存しない
2. **採用候補が勝った場合もスコアの下限を課す**。この処理は 0 件のときに
   しか走らないため、暗い・ぼやけている等の難しい画像に偏る。
   相対比較だけだと消去法で採用候補が勝ってしまう

ここで作るのは **画像全体を覆うボックス**なので、誤るとトラッキングと
点群生成を大きく歪める。保守的に倒している。

閾値未満だったものも **論理削除として残す**（スコア付き）。
UI の Show deleted + Box text=Score で分布を見て閾値を決められる。
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)

# 画像全体のボックスであることを示す印
DETECTION_LABEL = "whole_image"
DELETED_BY_LOW_SCORE = "whole_image_low_score"
DELETED_BY_REJECTED = "whole_image_rejected"


def resize_for_classification(
    image: Image.Image, ratio: float
) -> Image.Image:
    """分類へ渡すために縮小する.

    元解像度のままでは大きすぎる（SigLIP2 は内部でパッチ数の上限まで
    縮めるが、無駄に大きい画像を渡すと前処理が重い）。
    """
    if ratio >= 1.0 or ratio <= 0:
        return image
    width = max(1, int(image.width * ratio))
    height = max(1, int(image.height * ratio))
    return image.resize((width, height), Image.BILINEAR)


def classify_whole_image(
    classifier: Any,
    image: Image.Image,
    candidates: dict[str, str | None],
    *,
    score_threshold: float,
    resize_ratio: float = 0.5,
) -> dict[str, Any] | None:
    """画像全体を分類し、ボックス 1 件分の辞書を返す.

    Args:
        candidates: ``{分類候補: 最終ラベル or None}``。
            None は除外候補（判定されても検出としない）

    Returns:
        ボックスの辞書。採用なら ``is_deleted=False``、
        除外候補や低スコアなら ``is_deleted=True``（スコアは残す）。
        候補が無い場合は None。
    """
    if not candidates:
        return None

    candidate_labels = list(candidates)
    try:
        predicted, score = classifier.classify_with_score(
            resize_for_classification(image, resize_ratio), candidate_labels
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("whole-image classification failed: %s", exc)
        return None

    if predicted is None:
        return None

    final_label = candidates.get(predicted)
    box: dict[str, Any] = {
        # 画像全体を覆う。座標は **元の解像度**で返す
        "xmin": 0.0, "ymin": 0.0,
        "xmax": float(image.width), "ymax": float(image.height),
        "label": final_label or predicted,
        "detection_label": DETECTION_LABEL,
        "reclassified_as": predicted,
        "score": round(float(score), 4),
        "is_deleted": False,
        "deleted_by": None,
    }

    if final_label is None:
        # 除外候補。検出としては使わないが、確認できるよう残す
        box["is_deleted"] = True
        box["deleted_by"] = DELETED_BY_REJECTED
    elif score < score_threshold:
        # 採用候補だがスコアが足りない。閾値の較正に使えるよう残す
        box["is_deleted"] = True
        box["deleted_by"] = DELETED_BY_LOW_SCORE

    return box
