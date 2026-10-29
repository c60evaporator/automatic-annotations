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

    # "cuda" / "cpu"。未指定なら起動時に自動判定する
    DEVICE: str | None = None

    # --- Grounding DINO（Transformers 版）---------------------------------
    # HuggingFace のモデル ID。grounding-dino-base は公式 groundingdino_swinb_cogcoor に相当。
    # 重みは初回ロード時に HF_HOME へ自動ダウンロードされる（手動配置は不要）
    GROUNDINGDINO_MODEL: str = "IDEA-Research/grounding-dino-base"

    # --- SAM2（Transformers 版）------------------------------------------
    # HuggingFace のモデル ID。重みは初回ロード時に HF_HOME へ自動ダウンロードされる
    SAM2_MODEL: str = "facebook/sam2.1-hiera-large"

    # --- SigLIP2（ラベル再判定）--------------------------------------------
    # transformers 実装を使うので Dockerfile の変更は不要。
    # 初回実行時に HuggingFace からのダウンロードが走る（数 GB）
    SIGLIP2_MODEL: str = "google/siglip2-so400m-patch16-naflex"

    # --- Depth-Anything-3（公式リポジトリ版）-------------------------------
    # 重みは HuggingFace（depth-anything/<モデル名>）から HF_HOME へ自動ダウンロードされる。
    # DA3METRIC-LARGE / DA3NESTED-GIANT-LARGE(-1.1) に対応（da3_depth.get_metric_depth）
    DEPTH_ANYTHING_MODEL: str = "DA3METRIC-LARGE"
    # これ以遠の点は点群にしない [m]。遠方は誤差が大きく、
    # 入れるとクラスタリングが引きずられる
    DEPTH_MAX_DISTANCE: float = 60.0

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
