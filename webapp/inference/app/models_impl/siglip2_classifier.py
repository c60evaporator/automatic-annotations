"""SigLIP2 による zero-shot 分類（ラベル再判定）.

GroundingDINO はプロンプト由来の取り違えが多く、柵を barrier と拾ったり、
窓の反射を car と拾ったりする。ボックス周辺を切り出して zero-shot 分類に
かけ直すことで、こうした誤りを落とす。

transformers の実装を使うので、Dockerfile の変更は不要。
"""
from __future__ import annotations

from typing import Any, Sequence

from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)

# 分類に渡すプロンプトの型。SigLIP は学習時の形に近いほうが当たる
PROMPT_TEMPLATE = "This is a photo of {label}."
# processor に渡す固定長。SigLIP2 の既定に合わせる
TEXT_MAX_LENGTH = 64
MAX_NUM_PATCHES = 256


class Siglip2Classifier:
    def __init__(self, model_name: str, device: str = "cuda") -> None:
        # 重い import はここで行う。モジュールのトップに置くと
        # サーバー起動時に transformers と torch が読み込まれる
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.device = device
        logger.info("loading SigLIP2: %s", model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.info("SigLIP2 ready on %s", device)

    def to(self, device: str):
        """models.py の解放処理から呼ばれる（VRAM を返すため）."""
        self.model.to(device)
        self.device = device
        return self

    def classify_batch(
        self,
        images: Sequence[Image.Image],
        candidate_labels: Sequence[str],
        *,
        score_threshold: float | None = None,
    ) -> list[str | None]:
        """画像ごとに最も近い候補ラベルを返す.

        Args:
            score_threshold: 指定するとスコアがこれ未満の画像は None を返す

        Returns:
            画像と同じ長さのリスト。判定できなかった要素は None。

        NOTE: 参照実装は閾値未満の要素をリストから落としていたが、
        それだと入力とのインデックス対応が崩れる。
        ここでは None を入れて長さを保つ。
        """
        import torch

        if not images or not candidate_labels:
            return [None] * len(images)

        texts = [PROMPT_TEMPLATE.format(label=label) for label in candidate_labels]
        inputs = self.processor(
            text=texts,
            images=list(images),
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            truncation=True,
            max_num_patches=MAX_NUM_PATCHES,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # SigLIP は softmax ではなく sigmoid で各ペアの一致度を出す
        scores = torch.sigmoid(outputs.logits_per_image)
        best = scores.argmax(dim=1)

        results: list[str | None] = []
        for row, index in enumerate(best.tolist()):
            if (score_threshold is not None
                    and float(scores[row, index]) < score_threshold):
                results.append(None)
            else:
                results.append(candidate_labels[index])
        return results

    def classify(
        self,
        image: Image.Image,
        candidate_labels: Sequence[str],
        *,
        score_threshold: float | None = None,
    ) -> str | None:
        return self.classify_batch(
            [image], candidate_labels, score_threshold=score_threshold
        )[0]
