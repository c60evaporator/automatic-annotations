"""検出ボックスのラベル再判定.

GroundingDINO が付けたラベルごとに候補を切り替えて zero-shot 分類し、
対応表に従って最終ラベルを決める。対応表の値が None の候補（「使わない」候補）が
選ばれたボックスは、論理削除の対象として印を付ける。

削除は「使わない」候補の最大スコアが、残す候補の最大スコアより delete_margin
以上高いときだけ行う（SigLIP2 の sigmoid スコアの差）。0 なら単純にスコア最大の
候補で決める。取り違えで正しい検出を消すと見逃しになるため、確信が高いときだけ消す。

行そのものは消さない。UI で「どれが再判定で落ちたか」を確認したいので、
削除は is_deleted のフラグで表す。
"""
from __future__ import annotations

from typing import Any, Sequence

from PIL import Image

from app.core.logging import get_logger
from app.services.crop import crop_image_with_margin

logger = get_logger(__name__)


def reclassify_boxes(
    classifier: Any,
    image: Image.Image,
    boxes: Sequence[dict[str, Any]],
    candidates: dict[str, dict[str, str | None]],
    *,
    margin_ratio: float = 0.1,
    delete_margin: float = 0.0,
    keep_scores: bool = False,
) -> int:
    """1 フレーム分のボックスを再判定し、label / is_deleted を書き換える.

    Args:
        candidates: ``{検出ラベル: {分類候補: 最終ラベル or None}}``
        margin_ratio: 切り出し時にボックスを広げる比率
        delete_margin: 「使わない」候補の最大スコアが残す候補の最大スコアを
            この値より大きく上回ったときだけ削除する（0 ならスコア最大の候補で決める）
        keep_scores: True なら全候補のスコアを ``reclassification_scores`` に残す
            （閾値探索用のキャッシュで、判定の規則を後から変えて評価するため）

    Returns:
        再判定したボックス数。

    ボックスは **その場で書き換える**（呼び出し側が結果をそのまま使う）。
    ``detection_label`` に再判定前のラベルを残す。

    同じ検出ラベルのボックスはまとめて 1 回の推論にかける。
    1 個ずつ呼ぶと、候補テキストの前処理が毎回走って無駄が大きい。
    """
    if not boxes or not candidates:
        return 0

    # 検出ラベルごとにまとめる（候補リストがラベルごとに違うため）
    by_label: dict[str, list[dict[str, Any]]] = {}
    for box in boxes:
        # 再判定前のラベルを記録しておく。
        # NOTE: setdefault は使えない。スキーマ経由の dict には
        # detection_label キーが None 付きで既に存在するため、
        # 値が入らず再判定の対象判定が全て外れる
        if not box.get("detection_label"):
            box["detection_label"] = box.get("label")
        if box.get("is_deleted") is None:
            box["is_deleted"] = False
        target = box.get("detection_label")
        if target in candidates:
            by_label.setdefault(target, []).append(box)

    reclassified = 0
    for target_label, targets in by_label.items():
        mapping = candidates[target_label]
        candidate_labels = list(mapping)
        crops = [
            crop_image_with_margin(image, box, margin_ratio=margin_ratio)
            for box in targets
        ]
        try:
            scores = classifier.score_batch(crops, candidate_labels)
        except Exception as exc:  # noqa: BLE001
            # 1 ラベル分の失敗で全体を止めない。元のラベルを残す
            logger.warning("re-classification failed for %s: %s", target_label, exc)
            continue

        keep_cands = [i for i, c in enumerate(candidate_labels) if mapping[c] is not None]
        drop_cands = [i for i, c in enumerate(candidate_labels) if mapping[c] is None]
        for box, row in zip(targets, scores):
            if keep_scores:
                box["reclassification_scores"] = dict(zip(candidate_labels, row))
            if not row:
                # 判定できなかったものは元のラベルのまま残す
                continue
            best_keep = max(keep_cands, key=row.__getitem__, default=None)
            best_drop = max(drop_cands, key=row.__getitem__, default=None)
            if best_drop is not None and (
                best_keep is None or row[best_drop] > row[best_keep] + delete_margin
            ):
                # 「使わない」候補の確信が十分高い = このボックスは使わない（論理削除）
                box["reclassified_as"] = candidate_labels[best_drop]
                box["is_deleted"] = True
                box["deleted_by"] = "reclassification"
            else:
                box["reclassified_as"] = candidate_labels[best_keep]
                box["label"] = mapping[candidate_labels[best_keep]]
                box["is_deleted"] = False
            reclassified += 1

    return reclassified
