"""推論サーバーの設定."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# サードパーティのリポジトリは Dockerfile で /opt/third_party に clone する
THIRD_PARTY_ROOT = Path("/opt/third_party")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )

    APP_NAME:  str = "automatic-annotation-inference"
    LOG_LEVEL: str = "INFO"

    # webapp と同じマウント構成
    DATA_ROOT:    Path = Path("/data")
    DERIVED_ROOT: Path = Path("/derived")

    # 手動で配置した重みの置き場（compose で ./checkpoints をマウント）
    CHECKPOINT_DIR: Path = Path("/opt/checkpoints")

    # "cuda" / "cpu"。未指定なら起動時に自動判定する
    DEVICE: str | None = None

    # --- Grounding DINO（公式リポジトリ版）--------------------------------
    # HuggingFace 版ではなく IDEA-Research/GroundingDINO を使う。
    # config はリポジトリ同梱のものを、重みは手動配置したものを指す
    GROUNDINGDINO_CONFIG_PATH: Path = (
        THIRD_PARTY_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    )
    GROUNDINGDINO_WEIGHT_PATH: Path = Path(
        "/opt/checkpoints/groundingdino_swinb_cogcoor.pth"
    )

    # --- SAM2（公式リポジトリ版）------------------------------------------
    # config は Hydra の設定名（SAM2 パッケージ内の configs/ から解決される）。
    # ファイルパスではない点に注意
    SAM2_CONFIG_PATH: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    SAM2_CHECKPOINT_PATH: Path = Path("/opt/checkpoints/sam2.1_hiera_large.pt")

    # --- Depth-Anything-3（未実装）----------------------------------------
    DEPTH_ANYTHING_MODEL: str = "depth-anything/DA3-large"

    # GPU や重みが無い環境で UI を動かすためのスタブ切り替え
    USE_STUB_MODELS: bool = False

    # 同時に GPU へ載せるモデル数の上限。
    # GroundingDINO + SAM2 + DA3 を同時常駐させると VRAM が足りなくなる
    # 構成があるため、既定では 1 つずつロード・解放する
    MAX_RESIDENT_MODELS: int = 1

    # 完了したジョブを保持する時間（秒）
    JOB_RETENTION_SEC: int = 3600

    @property
    def device(self) -> str:
        """実際に使うデバイス.

        DEVICE 未指定なら CUDA の有無で決める。
        torch の import をここに閉じ込めてあるので、
        設定の読み込み自体は torch 無しでも通る。
        """
        if self.DEVICE:
            return self.DEVICE
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """設定のシングルトン.

    webapp 側と同様、モジュールレベルで Settings() を評価しない。
    環境変数が1つ欠けただけで import 自体が失敗し、原因が追いにくくなる。
    """
    return Settings()
