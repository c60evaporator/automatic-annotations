"""NuScenes 本体データセット JSON の Pydantic スキーマ.

読み込み方（TypeAdapter で一括バリデーションすると trainval でも実用速度が出る）:

    from pydantic import TypeAdapter
    import orjson

    with open(path, "rb") as f:
        raw = orjson.loads(f.read())
    records = TypeAdapter(list[SampleAnnotation]).validate_python(raw)
"""
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict

# 空文字列 "" または null を None に変換するカスタム型（任意FK用）
OptionalToken = Annotated[str | None, BeforeValidator(lambda v: None if v == "" else (str(v) if v else None))]

# 空リスト [] → None（CalibratedSensor.camera_intrinsic 用）
def _empty_list_to_none(v):
    return None if v == [] else v

# 0 → None（SampleData.height / width 用：非カメラセンサーは 0 で記録される）
def _zero_to_none(v):
    return None if v == 0 else v


class NuScenesRecord(BaseModel):
    """全スキーマ共通の基底クラス.

    extra='ignore' を全体に効かせる理由:
    nuScenes は版によって JSON のフィールドが増減する
    （例: category.json の `index` は一部の版にのみ存在し、
      v1.0-mini と v1.0-trainval でも差がある）。
    未知フィールドで落とすとバージョン差で毎回インポートが止まるため、
    DB モデルに無いものは黙って捨てる。
    """
    model_config = ConfigDict(extra="ignore")


# ── Log ───────────────────────────────────────────────────────────────────────

class Log(NuScenesRecord):
    token: str
    logfile: str
    vehicle: str
    date_captured: str
    location: str  # 'boston-seaport', 'singapore-onenorth' etc.


# ── Scene ─────────────────────────────────────────────────────────────────────

class Scene(NuScenesRecord):
    token: str
    log_token: str
    nbr_samples: int
    first_sample_token: str
    last_sample_token: str
    name: str
    description: str | None = None


# ── Sample ────────────────────────────────────────────────────────────────────

class Sample(NuScenesRecord):
    token: str
    timestamp: int
    prev: OptionalToken  # 先頭サンプルは ""
    next: OptionalToken  # 末尾サンプルは ""
    scene_token: str


# ── Category ──────────────────────────────────────────────────────────────────

class Category(NuScenesRecord):
    token: str
    name: str
    description: str | None = None


# ── Attribute ─────────────────────────────────────────────────────────────────

class Attribute(NuScenesRecord):
    token: str
    name: str
    description: str | None = None


# ── Visibility ────────────────────────────────────────────────────────────────

class Visibility(NuScenesRecord):
    # token は "1"〜"4" で UUID ではない。
    # 別データセットを同一 DB に読むと衝突するが、CLAUDE.md の
    # 「1 データセット 1 メタデータ」制約で運用回避する
    token: str
    level: str
    description: str | None = None


# ── Instance ──────────────────────────────────────────────────────────────────

class Instance(NuScenesRecord):
    token: str
    category_token: str
    nbr_annotations: int
    # 実データでは常に埋まっているが、DB モデル側が nullable なので
    # 型も合わせて OptionalToken にしておく（"" が来ても落ちない）
    first_annotation_token: OptionalToken
    last_annotation_token: OptionalToken


# ── SampleAnnotation ──────────────────────────────────────────────────────────

class SampleAnnotation(NuScenesRecord):
    token: str
    sample_token: str
    instance_token: str
    visibility_token: OptionalToken
    attribute_tokens: list[str]
    translation: list[float]  # [x, y, z]
    size: list[float]         # [width, length, height]
    rotation: list[float]     # [w, x, y, z]
    prev: OptionalToken  # 先頭アノテーションは ""
    next: OptionalToken  # 末尾アノテーションは ""
    num_lidar_pts: int
    num_radar_pts: int


# ── Sensor ────────────────────────────────────────────────────────────────────

class Sensor(NuScenesRecord):
    token: str
    channel: str   # 'CAM_FRONT', 'LIDAR_TOP' etc.
    modality: str  # 'camera', 'lidar', 'radar'


# ── CalibratedSensor ──────────────────────────────────────────────────────────

class CalibratedSensor(NuScenesRecord):
    token: str
    sensor_token: str
    translation: list[float]  # [x, y, z]
    rotation: list[float]     # [w, x, y, z]
    # カメラのみ 3x3 行列、非カメラは [] → None
    camera_intrinsic: Annotated[
        list[list[float]] | None,
        BeforeValidator(_empty_list_to_none),
    ]


# ── EgoPose ───────────────────────────────────────────────────────────────────

class EgoPose(NuScenesRecord):
    token: str
    timestamp: int
    translation: list[float]  # [x, y, z]
    rotation: list[float]     # [w, x, y, z]


# ── SampleData ────────────────────────────────────────────────────────────────

class SampleData(NuScenesRecord):
    token: str
    sample_token: str
    ego_pose_token: str
    calibrated_sensor_token: str
    timestamp: int
    fileformat: str   # 'jpg', 'pcd', 'bin', 'npz'
    is_key_frame: bool
    # カメラのみ非ゼロ。非カメラは 0 → None
    height: Annotated[int | None, BeforeValidator(_zero_to_none)]
    width: Annotated[int | None, BeforeValidator(_zero_to_none)]
    filename: str
    prev: OptionalToken  # 先頭フレームは ""
    next: OptionalToken  # 末尾フレームは ""


# ── Map ───────────────────────────────────────────────────────────────────────

class Map(NuScenesRecord):
    """map.json の1レコード（basemap 画像への参照）.

    MapMeta モデルを埋めるには、この JSON だけでは足りない点に注意:
      - location   : log_tokens から Log を引いて log.location を取得する
                     （1つの map は同一 location の複数 log から参照される）
      - version    : maps/expansion/<location>.json の "version"
      - canvas_edge: 同上の "canvas_edge"
    つまり map.json + log.json + expansion JSON の3つを突き合わせる。
    """
    token: str
    filename: str          # 'maps/xxxx.png'（basemap_path に入れる）
    category: str          # 'semantic_prior'
    log_tokens: list[str]


class MapExpansion(NuScenesRecord):
    """maps/expansion/<location>.json のトップレベル.

    ポリゴン・レーン等の巨大な配列は Map Expansion アノテーション機能を
    実装する段階で扱う。ここでは MapMeta を埋めるのに必要な2つだけ拾う
    （extra='ignore' なので残りは自動的に捨てられ、メモリも食わない）。
    """
    version: str                 # '1.3' etc.
    canvas_edge: list[float]     # [width_m, height_m]
