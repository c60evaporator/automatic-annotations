@dataclass
class Box2D:
    """2D バウンディングボックス 1 個（画像平面）。

    Attributes:
        xyxy: shape ``(4,)``、``np.float64``。``(x0, y0, x1, y1)``[px]。原点は左上、
            ``x`` 右・``y`` 下（`2d_tasks.md` 3.2）。フィールド名を ``xyxy`` にすることで
            xywh との取り違えをコード上で防ぐ（``Box3D.size`` の並び順と同じ発想）。
        label: クラス名の**文字列**。ゼロショット分類・オープン語彙検出では
            ラベル集合が実行時に決まるためクラスインデックスにできない
        score: 信頼度スコア ``[0, 1]``。GTでは ``None``。
        track_id: 追跡 ID。単発検出では ``None``。
        attributes: データセット固有の属性。
    """

    xyxy: np.ndarray
    label: str | None = None
    score: float | None = None
    track_id: int | None = None
    attributes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.xyxy.shape != (4,):
            raise ValueError(f"Box2D.xyxy must have shape (4,), got {self.xyxy.shape}")

    def convert_to_original_coordinates(
        self,
        original_width: int,
        original_height: int,
        crop_xyxy: tuple[int, int, int, int] | None = None,
        input_width: float | None = None,
        input_height: float | None = None,
        normalized: bool = True,
    ) -> Box2D:
        """
        ボックスを元画像座標系に変換する。以下の順で変換された前提とする

        original_image_size → crop_xyxyの範囲で切出 → input_width/heightにリサイズ → normalizedの範囲[0, 1]に正規化(normalized=Trueの場合)

        Args:
            original_width: 元画像の幅[px]。
            original_height: 元画像の高さ[px]。
            crop_xyxy: crop した場合の crop 前の元画像座標系での切り出し範囲。None の場合は crop なし。
            input_width: モデル入力時の画像幅[px]。None の場合はリサイズなし。
            input_height: モデル入力時の画像高さ[px]。None の場合はリサイズなし。
            normalized: True の場合は ``xyxy`` が ``[0, 1]`` の正規化座標であるとみなし、元画像座標に戻す
        """
        if crop_xyxy is None:
            crop_xyxy = (0, 0, original_width, original_height)
        if input_width is None:
            input_width = original_width
        if input_height is None:
            input_height = original_height

        # 正規化座標をモデル入力画像のピクセル座標に変換
        if normalized:
            xyxy = self.xyxy * np.array([input_width, input_height, input_width, input_height], dtype=np.float64)
        else:
            xyxy = self.xyxy.astype(np.float64)
        # 元画像座標系に戻す
        xyxy = rect_to_original(
            xyxy=xyxy,
            original_width=original_width,
            original_height=original_height,
            crop_xyxy=crop_xyxy,
            resized_width=input_width,
            resized_height=input_height,
        )

        return Box2D(xyxy=xyxy, label=self.label, score=self.score, track_id=self.track_id, attributes=self.attributes)
