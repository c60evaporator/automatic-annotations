"""FastAPI アプリの初期化とルーター登録.

main.py の方針:

  - **APIRouter を使う。** @app.get を main.py に直接書くと、
    3ステップ分のエンドポイントが1ファイルに集まって見通しが悪くなる。
    ルーターごとに prefix と tags を付ければ /docs も自動で分類される。

  - **モデルは lifespan でロードしない。** 起動時に3モデルを載せると
    VRAM が厳しく、起動も遅い。lifespan ではローダーの「登録」だけ行い、
    実体は最初のリクエストで載せる（app/core/models.py）。

  - **on_event ではなく lifespan を使う。** @app.on_event は非推奨。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.jobs import get_job_manager
from app.core.logging import get_logger, setup_logging
from app.core import models as model_registry
from app.routers import depth_boxfitting, det2d, instance_tracking

logger = get_logger(__name__)


def register_model_loaders() -> None:
    """モデルのロード関数を登録する（この時点では実行しない）.

    重いライブラリの import はローダーの内側で行う。
    ここでトップレベル import すると、1つでも壊れているだけで
    サーバー自体が起動しなくなり、原因の切り分けが難しくなる。
    """
    settings = get_settings()

    def load_grounding_dino():
        from app.models_impl.grounding_dino import GroundingDinoDetector
        return GroundingDinoDetector(
            model_id=settings.GROUNDING_DINO_MODEL, device=settings.DEVICE
        )

    def load_sam2():
        from app.models_impl.sam2_tracker import Sam2Tracker
        return Sam2Tracker(model_id=settings.SAM2_MODEL, device=settings.DEVICE)

    def load_depth_anything():
        from app.models_impl.depth_anything import DepthAnythingEstimator
        return DepthAnythingEstimator(
            model_id=settings.DEPTH_ANYTHING_MODEL, device=settings.DEVICE
        )

    model_registry.register_loader("grounding_dino", load_grounding_dino)
    model_registry.register_loader("sam2", load_sam2)
    model_registry.register_loader("depth_anything", load_depth_anything)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger.info("starting %s (device=%s)", settings.APP_NAME, settings.DEVICE)
    register_model_loaders()
    try:
        yield
    finally:
        logger.info("shutting down")
        get_job_manager().shutdown()
        model_registry.unload_all()


app = FastAPI(
    title="Automatic Annotation Inference API",
    description=(
        "2D Object Detection → Instance Tracking → Depth Estimation & Box Fitting "
        "のパイプラインを提供する推論サーバー"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(det2d.router)
app.include_router(instance_tracking.router)
app.include_router(depth_boxfitting.router)


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    """ヘルスチェック.

    モデルのロードは待たない。ここでロード完了を条件にすると、
    compose の healthcheck が数分間 unhealthy のままになり、
    webapp の起動が止まる。
    """
    return {"status": "ok", "loaded_models": model_registry.loaded_models()}


@app.get("/system/info", tags=["system"])
def system_info() -> dict[str, Any]:
    """GPU とモデルの状態を返す（UI の表示・デバッグ用）."""
    settings = get_settings()
    info: dict[str, Any] = {
        "device": settings.DEVICE,
        "max_resident_models": settings.MAX_RESIDENT_MODELS,
        "loaded_models": model_registry.loaded_models(),
        "cuda": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["cuda"] = {
                "device_name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(total / 1024**2),
                "vram_free_mb": round(free / 1024**2),
                "torch_version": torch.__version__,
            }
    except ImportError:
        pass
    return info


@app.post("/system/unload", tags=["system"])
def unload_models() -> dict[str, Any]:
    """常駐モデルを解放して VRAM を返す（手動運用・デバッグ用）."""
    before = model_registry.loaded_models()
    model_registry.unload_all()
    return {"unloaded": before}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    """未捕捉の例外を JSON で返す.

    推論中の例外はスタックトレースが長くなりがちなので、
    クライアントには型とメッセージだけを返し、詳細はログに残す。
    """
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )
