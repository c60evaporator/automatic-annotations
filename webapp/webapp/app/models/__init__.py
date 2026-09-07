"""SQLAlchemy ORM モデル（スキーマの唯一の正）.

Alembic が全モデルを検出できるよう、ここで全モデルを import する。
import 順は「参照される側 → 参照する側」に揃えてあるが、
relationship は全て文字列のフォワード参照なので実行順序に依存しない。
"""
from app.db.base import Base

from app.models.dataset import Dataset
from app.models.scene import Log, Sample, Scene
from app.models.sensor import CalibratedSensor, EgoPose, SampleData, Sensor
from app.models.annotation import (
    SOURCE_AUTO,
    SOURCE_IMPORTED,
    SOURCE_MANUAL,
    Attribute,
    Category,
    Instance,
    SampleAnnotation,
    Visibility,
    annotation_attribute,
)
from app.models.map import MapMeta
from app.models.ann_intermediate import (
    BOXFIT_STATUS_FAILED,
    BOXFIT_STATUS_FITTED,
    BOXFIT_STATUS_NO_POINTS,
    BOXFIT_STATUS_TOO_FEW_POINTS,
    BOXFIT_STATUSES,
    INSTANCE_ORIGIN_PROMPT,
    INSTANCE_ORIGIN_PROPAGATED,
    INSTANCE_ORIGINS,
    IOU_LABEL_MATCH_CATEGORY_GROUP,
    IOU_LABEL_MATCH_LABEL,
    IOU_LABEL_MATCH_NONE,
    IOU_LABEL_MATCHES,
    IOU_METHOD_BOX,
    IOU_METHOD_MASK,
    IOU_METHODS,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    RUN_STATUSES,
    BoxFitting3D,
    DepthEstimation,
    DepthEstimationParams,
    LidarPointcloud,
    Detection2D,
    Detection2DParams,
    InstanceTracking2D,
    InstanceTracking2DParams,
)

__all__ = [
    "Base",
    # dataset
    "Dataset",
    # scene
    "Log",
    "Scene",
    "Sample",
    # sensor
    "Sensor",
    "CalibratedSensor",
    "EgoPose",
    "SampleData",
    # annotation
    "Category",
    "Attribute",
    "Visibility",
    "Instance",
    "SampleAnnotation",
    "annotation_attribute",
    "SOURCE_IMPORTED",
    "SOURCE_AUTO",
    "SOURCE_MANUAL",
    # map
    "MapMeta",
    # ann_intermediate
    "Detection2DParams",
    "Detection2D",
    "InstanceTracking2DParams",
    "InstanceTracking2D",
    "DepthEstimationParams",
    "DepthEstimation",
    "RUN_STATUS_RUNNING",
    "RUN_STATUS_SUCCEEDED",
    "RUN_STATUS_FAILED",
    "RUN_STATUS_CANCELLED",
    "RUN_STATUSES",
    "IOU_METHOD_BOX",
    "IOU_METHOD_MASK",
    "IOU_METHODS",
    "IOU_LABEL_MATCH_LABEL",
    "IOU_LABEL_MATCH_CATEGORY_GROUP",
    "IOU_LABEL_MATCH_NONE",
    "IOU_LABEL_MATCHES",
    "INSTANCE_ORIGIN_PROMPT",
    "INSTANCE_ORIGIN_PROPAGATED",
    "INSTANCE_ORIGINS",
    "LidarPointcloud",
    "BoxFitting3D",
    "BOXFIT_STATUS_FITTED",
    "BOXFIT_STATUS_TOO_FEW_POINTS",
    "BOXFIT_STATUS_NO_POINTS",
    "BOXFIT_STATUS_FAILED",
    "BOXFIT_STATUSES",
]
