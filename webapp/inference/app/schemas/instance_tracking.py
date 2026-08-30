@dataclass
class Instance2D:
    """インスタンスセグメンテーションの 1 インスタンス。

    マスクは**全画面ではなく bbox 内に限定**して持つ。素朴な ``(N, H, W)`` の全画面
    マスクは 1600x900 で 50 インスタンスなら約 72MB になり、プロセス境界を越える設計では
    実用にならないため。

    Attributes:
        box: このインスタンスのバウンディングボックス。
        mask: shape ``(h, w)``、``np.bool_``。``mask_region`` が示す整数画素領域を覆う。
        mask_region: ``(x0, y0, x1, y1)`` の**整数**画素領域。``x1 - x0 == w``、
            ``y1 - y0 == h`` を満たす。``box.xyxy``（float）とは別に整数領域を持つのは、
            float から暗黙に丸めると実装ごとに 1 画素ずれるため（曖昧さの排除）。
    """

    box: Box2D
    mask: np.ndarray | None = None
    mask_region: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if (self.mask is None) != (self.mask_region is None):
            raise ValueError("Instance2D.mask and mask_region must be provided together")
        if self.mask is not None:
            if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
                raise ValueError(
                    "Instance2D.mask must be a 2-D bool array, "
                    f"got shape={self.mask.shape} dtype={self.mask.dtype}"
                )
            assert self.mask_region is not None  # 上の同時指定チェックで保証済み
            x0, y0, x1, y1 = self.mask_region
            height, width = self.mask.shape
            if (x1 - x0, y1 - y0) != (width, height):
                raise ValueError(
                    "Instance2D.mask_region size must match mask shape "
                    f"(expected x1-x0={width}, y1-y0={height}; got {(x1 - x0, y1 - y0)})"
                )

    def convert_to_original_coordinates(
            self,
            original_width: int,
            original_height: int,
            crop_xyxy: tuple[int, int, int, int] | None = None,
            input_width: float | None = None,
            input_height: float | None = None,
            normalized: bool = False,
        ) -> Instance2D:
            """
            インスタンスを元画像座標系に変換する。以下の順で変換された前提とする
    
            original_image_size → crop_xyxyの範囲で切出 → resize_scale_x/yでリサイズ → normalizedの範囲[0, 1]に正規化(normalized=Trueの場合)
    
            Args:
                original_width: 元画像の幅[px]。
                original_height: 元画像の高さ[px]。
                crop_xyxy: crop した場合の crop 前の元画像座標系での切り出し範囲。None の場合は crop なし。
                input_width: モデル入力時の画像幅[px]。None の場合はリサイズなし。
                input_height: モデル入力時の画像高さ[px]。None の場合はリサイズなし。
                normalized: True の場合は ``box.xyxy`` が ``[0, 1]`` の正規化座標であるとみなし、元画像座標に戻す
            """
            if self.mask is None:
                raise ValueError("Instance2D.mask is None, cannot convert to original coordinates")
            
            if self.mask_region is None:
                raise ValueError("Instance2D.mask_region is None, cannot convert to original coordinates")

            # boxを元画像座標系に変換
            new_box = self.box.convert_to_original_coordinates(
                original_width=original_width,
                original_height=original_height,
                crop_xyxy=crop_xyxy,
                input_width=input_width,
                input_height=input_height,
                normalized=normalized,
            )

            # mask_regionを元画像座標系に変換
            mask_region_array = np.array(self.mask_region, dtype=np.float64)
            moved_mask_region = rect_to_original(
                xyxy=mask_region_array,
                original_width=original_width,
                original_height=original_height,
                crop_xyxy=crop_xyxy,
                resized_width=input_width,
                resized_height=input_height,
            )
            # 先に整数の目標領域を確定し、そのサイズへマスクを合わせる（逆順にすると
            # 丸めで 1 画素ずれ、Instance2D の size 一致検証に落ちる）。
            x0, y0 = int(np.floor(moved_mask_region[0])), int(np.floor(moved_mask_region[1]))
            x1, y1 = int(np.ceil(moved_mask_region[2])), int(np.ceil(moved_mask_region[3]))
            x1 = max(x1, x0 + 1)  # 極端な縮小で幅・高さが 0 になるのを防ぐ。
            y1 = max(y1, y0 + 1)
            new_mask_region = (x0, y0, x1, y1)

            # maskにリサイズ逆変換を適用
            if (y1 - y0, x1 - x0) != self.mask.shape:
                new_mask = resize_mask_nearest(mask=self.mask, height=y1 - y0, width=x1 - x0)
            else:
                new_mask = self.mask.copy()

            return Instance2D(box=new_box, mask=new_mask, mask_region=new_mask_region)
