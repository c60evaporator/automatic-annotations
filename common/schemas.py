from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .geometry import quaternion_to_yaw, rect_to_original, resize_mask_nearest

@dataclass
class Box3D:
    """3D バウンディングボックス 1 個。

    Attributes:
        center: shape ``(3,)``、``np.float64``。ボックス中心の座標[m]。
            **底面中心ではなく重心（幾何中心）**である点に注意（nuScenes 準拠）。
        size: shape ``(3,)``、``np.float64``。``(length, width, height)``[m] の順で、
            ego 座標系の x / y / z 軸方向の寸法に対応する。
            **nuScenes の ``sample_annotation.size`` は ``(width, length, height)`` 順**で
            あり、順番の入れ替えが必要（取り違えても例外は出ないため注意）。
            個別の値には ``size[0]`` 等でインデックスせず、
            :attr:`length` / :attr:`width` / :attr:`height` プロパティを使うこと。
        rotation: shape ``(4,)``、``np.float64``。``(w, x, y, z)`` 順のクォータニオン。
        label: クラス名の文字列（例: ``"car"``）。
        score: 信頼度スコア ``[0, 1]``。GT の場合は ``None``。
        velocity: shape ``(2,)`` または ``(3,)``、``np.float64``。速度[m/s]。
            ``frame`` と同じ座標系で表す。
        track_id: 追跡 ID。単体検出では ``None``。
        attributes: データセット固有の属性（nuScenes の ``vehicle.moving`` 等）。
    """

    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    label: str
    score: float | None = None
    velocity: np.ndarray | None = None
    track_id: int | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def yaw(self) -> float:
        """z 軸まわりの yaw 角[rad]（読み取り専用の派生値）。

        BEV 描画や yaw ベースのフレームワークとの連携用。
        **この値を正として保持しないこと**（情報が欠落するため）。
        """
        return quaternion_to_yaw(self.rotation)

    @property
    def length(self) -> float:
        """車長[m]。ego 座標系の x 軸方向の寸法（``size[0]``）。"""
        return float(self.size[0])

    @property
    def width(self) -> float:
        """車幅[m]。ego 座標系の y 軸方向の寸法（``size[1]``）。"""
        return float(self.size[1])

    @property
    def height(self) -> float:
        """車高[m]。ego 座標系の z 軸方向の寸法（``size[2]``）。"""
        return float(self.size[2])

    @classmethod
    def from_dimensions(
        cls,
        center: np.ndarray,
        length: float,
        width: float,
        height: float,
        rotation: np.ndarray,
        label: str,
        **kwargs: object,
    ) -> Box3D:
        """寸法をキーワードで指定して構築する。

        ``size`` の並び順を意識せずに済むため、BBox構築ではこちらを使うことを推奨する。
        nuScenes のように``(width, length, height)`` 順で寸法を保持する形式からの変換で、
        取り違えを防げる。

        ``score`` / ``velocity`` / ``track_id`` / ``attributes`` は ``kwargs`` で
        そのまま渡せる。
        """
        size = np.array([length, width, height], dtype=np.float64)
        return cls(center=center, size=size, rotation=rotation, label=label, **kwargs)  # type: ignore[arg-type]

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
