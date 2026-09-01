"""Grounding DINO のテキストプロンプト構築.

torch に依存させていない理由:
GroundingDINO はプロンプトの作り方に結果が強く左右されるため、
ここだけは GPU 無しでも単体テストできるようにしておきたい。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptDefinition:
    caption: str
    raw_labels: tuple[str, ...]
    character_spans: tuple[tuple[tuple[int, int], ...], ...]


def build_multi_label_prompt(labels: list[str]) -> PromptDefinition:
    """Create a prompt for Grounding DINO from a list of labels

    Args:
        labels: List of labels to detect in the image.

    Returns:
        A PromptDefinition object containing the caption, raw labels, and token spans.

        example:
        labels = ["cat", "dog", "bird"]
        returns:
        PromptDefinition(
            caption="cat. dog. bird.",
            raw_labels=("cat", "dog", "bird"),
            character_spans=(((0, 3),), ((5, 8),), ((10, 14),)),
        )
    """
    caption = ""
    raw_labels: list[str] = []
    character_spans: list[tuple[tuple[int, int], ...]] = []

    seen_labels: set[str] = set()
    prompt_to_canonical_label: dict[str, str] = {}

    for raw_label in labels:
        # lowercase, strip whitespace, and remove trailing periods
        canonical_label = raw_label.lower().strip().rstrip(".")

        # remove duplicate labels and empty labels
        if not canonical_label:
            continue
        if canonical_label in seen_labels:
            continue
        seen_labels.add(canonical_label)

        # replace underscores with spaces and normalize whitespace
        prompt_label = canonical_label.replace("_", " ")
        prompt_label = " ".join(prompt_label.split())

        # Check for conflicting labels that produce the same prompt phrase
        previous_label = prompt_to_canonical_label.get(prompt_label)
        if (
            previous_label is not None
            and previous_label != canonical_label
        ):
            raise ValueError(
                "Different labels produce the same prompt phrase: "
                f"{previous_label!r} and {canonical_label!r}"
            )
        prompt_to_canonical_label[prompt_label] = canonical_label

        # Add the space to the prompt if it's not the first label
        if caption:
            caption += " "

        # Store the start and end indices of the prompt label in the caption
        # to restore the original labels later
        start = len(caption)
        caption += prompt_label
        end = len(caption)
        character_spans.append(((start, end),))

        # Store the original label for later use
        raw_labels.append(canonical_label)
        caption += "."

    if not raw_labels:
        raise ValueError("At least one label is required.")

    return PromptDefinition(
        caption=caption,
        raw_labels=tuple(raw_labels),
        character_spans=tuple(character_spans),
    )


def denormalize_boxes(
    boxes_xyxy: list[list[float]], width: int, height: int
) -> list[tuple[int, int, int, int]]:
    """正規化座標 (0〜1) の xyxy を画素座標の整数に変換する.

    Grounding DINO は正規化座標で返すため、DB へ保存する前に
    画像サイズを掛けて整数化する必要がある。
    丸め後に xmin >= xmax にならないよう、最低1px の幅・高さを保証する。
    """
    out: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        xmin = int(round(x1 * width))
        ymin = int(round(y1 * height))
        xmax = int(round(x2 * width))
        ymax = int(round(y2 * height))
        # 画像外にはみ出さないようクランプ
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))
        # 極小ボックスが潰れて幅0になるのを防ぐ
        if xmax <= xmin:
            xmax = min(xmin + 1, width)
            xmin = max(0, xmax - 1)
        if ymax <= ymin:
            ymax = min(ymin + 1, height)
            ymin = max(0, ymax - 1)
        out.append((xmin, ymin, xmax, ymax))
    return out
