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


# ── Log ───────────────────────────────────────────────────────────────────────

class Log(BaseModel):
    token: str
    logfile: str
    vehicle: str
    date_captured: str
    location: str  # 'boston-seaport', 'singapore-onenorth' etc.


# ── Scene ─────────────────────────────────────────────────────────────────────

class Scene(BaseModel):
    token: str
    log_token: str
    nbr_samples: int
    first_sample_token: str
    last_sample_token: str
    name: str
    description: str | None = None


# ── Sample ────────────────────────────────────────────────────────────────────

class Sample(BaseModel):
    token: str
    timestamp: int
    prev: OptionalToken  # 先頭サンプルは ""
    next: OptionalToken  # 末尾サンプルは ""
    scene_token: str


# ── Category ──────────────────────────────────────────────────────────────────

class Category(BaseModel):
    model_config = ConfigDict(extra='ignore')  # JSON の `index` フィールドは DBモデルにないため無視

    token: str
    name: str
    description: str | None = None


# ── Attribute ─────────────────────────────────────────────────────────────────

class Attribute(BaseModel):
    token: str
    name: str
    description: str | None = None


# ── Visibility ────────────────────────────────────────────────────────────────

class Visibility(BaseModel):
    token: str  # "1"~"4"（UUID ではない）
    level: str
    description: str | None = None


# ── Instance ──────────────────────────────────────────────────────────────────

class Instance(BaseModel):
    token: str
    category_token: str
    nbr_annotations: int
    first_annotation_token: str
    last_annotation_token: str


# ── SampleAnnotation ──────────────────────────────────────────────────────────

class SampleAnnotation(BaseModel):
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

class Sensor(BaseModel):
    token: str
    channel: str   # 'CAM_FRONT', 'LIDAR_TOP' etc.
    modality: str  # 'camera', 'lidar', 'radar'


# ── CalibratedSensor ──────────────────────────────────────────────────────────

class CalibratedSensor(BaseModel):
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

class EgoPose(BaseModel):
    token: str
    timestamp: int
    translation: list[float]  # [x, y, z]
    rotation: list[float]     # [w, x, y, z]


# ── SampleData ────────────────────────────────────────────────────────────────

class SampleData(BaseModel):
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
