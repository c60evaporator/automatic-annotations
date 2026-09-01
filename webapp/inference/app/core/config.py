"""推論サーバーの設定."""
from functools import lru_cache
from pathlib import Path
 
from pydantic_settings import BaseSettings, SettingsConfigDict
 
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )
 
    APP_NAME:  str = "automatic-annotation-inference"
    LOG_LEVEL: str = "INFO"
 
    # webapp と同じマウント構成
    DATA_ROOT:    Path = Path("/data")
    DERIVED_ROOT: Path = Path("/derived")
 
    CHECKPOINT_DIR: Path = Path("/opt/checkpoints")
 
    DEVICE: str = "cuda"
 
    # モデル識別子（重みは HF_HOME / CHECKPOINT_DIR にキャッシュされる）
    GROUNDING_DINO_MODEL: str = "IDEA-Research/grounding-dino-base"
    SAM2_MODEL:           str = "facebook/sam2.1-hiera-large"
    DEPTH_ANYTHING_MODEL: str = "depth-anything/DA3-large"
 
    # 同時に GPU へ載せるモデル数の上限。
    # GroundingDINO + SAM2 + DA3 を同時常駐させると VRAM が足りなくなる
    # 構成があるため、既定では 1 つずつロード・解放する
    MAX_RESIDENT_MODELS: int = 1
 
    # 完了したジョブを保持する時間（秒）
    JOB_RETENTION_SEC: int = 3600
 
 
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """設定のシングルトン.
 
    webapp 側と同様、モジュールレベルで Settings() を評価しない。
    環境変数が1つ欠けただけで import 自体が失敗し、原因が追いにくくなる。
    """
    return Settings()
