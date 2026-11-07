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

    # "cuda" / "mps" / "cpu"。未指定なら起動時に自動判定する（cuda → mps → cpu の順）
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

    # --- 深度推定（Depth-Anything-3 / MapAnything。公式リポジトリ版）--------
    # 重みは HuggingFace から HF_HOME へ自動ダウンロードされる。
    # モデルはリクエストの depth_model で選ぶ。これは未指定のときの既定値
    DEFAULT_DEPTH_MODEL: str = "facebook/map-anything-apache"
    # これ以遠の点は点群にしない [m]。遠方は誤差が大きく、
    # 入れるとクラスタリングが引きずられる
    DEPTH_MAX_DISTANCE: float = 60.0

    # --- PartHOE（歩行者の向き推定。公式リポジトリ版）---------------------
    # Hugging Face に重みが無いため、初回ロード時に Google Drive から
    # CHECKPOINT_DIR / PARTHOE_CHECKPOINT へダウンロードする（compose で ./checkpoints をマウント）
    CHECKPOINT_DIR: Path = Path("/opt/checkpoints")
    PARTHOE_ROOT: Path = Path("/opt/third_party/Part_HOE")
    PARTHOE_CHECKPOINT: str = "part_hoe/parthoe_s.pth"
    PARTHOE_CHECKPOINT_URL: str = (
        "https://drive.usercontent.google.com/download"
        "?id=1_u9kN9C-bBPSYjsMDxpdzc7Q-lfCdR0J&export=download&confirm=t"
    )

    # --- Orient-Anything V1 Large（物体の向き推定）------------------------
    # 重みは初回ロード時に Hugging Face から HF_HOME へダウンロードされる（約 1.2GB）。
    # バックボーン（DINOv2-Large）は構成と前処理の設定だけ読む（重みはチェックポイントに含まれる）。
    # リビジョンを固定して、重みの差し替えで結果が変わらないようにする
    ORIENT_ANYTHING_REPO: str = "Viglong/Orient-Anything"
    ORIENT_ANYTHING_CHECKPOINT: str = "croplargeEX2/dino_weight.pt"
    ORIENT_ANYTHING_REVISION: str = "ecfa089828d9fab521d47133bedbdbfdef151cd4"
    ORIENT_ANYTHING_BACKBONE: str = "facebook/dinov2-large"
    ORIENT_ANYTHING_BACKBONE_REVISION: str = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"

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

        DEVICE 未指定なら CUDA → MPS（Apple Silicon）→ CPU の順で決める。
        torch の import をここに閉じ込めてあるので、
        設定の読み込み自体は torch 無しでも通る。
        """
        if self.DEVICE:
            return self.DEVICE
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """設定のシングルトン.

    webapp 側と同様、モジュールレベルで Settings() を評価しない。
    環境変数が1つ欠けただけで import 自体が失敗し、原因が追いにくくなる。
    """
    return Settings()
