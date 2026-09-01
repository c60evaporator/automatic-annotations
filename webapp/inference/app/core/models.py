"""モデルの遅延ロードと GPU 占有の管理.

方針:

1. **起動時に全モデルをロードしない。**
   GroundingDINO / SAM2 / Depth-Anything-3 を同時常駐させると VRAM が
   厳しい構成がある。最初のリクエストが来たときにロードし、
   MAX_RESIDENT_MODELS を超えたら古いものから解放する。

2. **GPU 実行は必ず直列化する。**
   GPU は1枚しかないので、複数リクエストが同時に走ると
   VRAM 不足かスループット低下のどちらかになる。
   gpu_lock で1本に絞る。

3. **ロードはロックの内側で1回だけ。**
   同時に2リクエストが来ても二重ロードしないよう二重チェックする。
"""
from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# GPU を使う処理を直列化するロック。
# エンドポイントは `def`（同期）で定義し、FastAPI のスレッドプールで
# 実行させたうえで、この中で1本に絞る
gpu_lock = threading.RLock()

_models: OrderedDict[str, Any] = OrderedDict()
_loaders: dict[str, Callable[[], Any]] = {}
_load_lock = threading.Lock()


def register_loader(name: str, loader: Callable[[], Any]) -> None:
    """モデルのロード関数を登録する.

    loader は実際に呼ばれるまで実行されない。
    重い import（GroundingDINO 等）は loader の中で行うこと。
    そうしないとサーバー起動時に全部 import され、
    1つでも壊れているとサーバー自体が起動しなくなる。
    """
    _loaders[name] = loader


def is_loaded(name: str) -> bool:
    return name in _models


def loaded_models() -> list[str]:
    return list(_models.keys())


def get_model(name: str) -> Any:
    """モデルを取得する（未ロードならロードする）."""
    model = _models.get(name)
    if model is not None:
        _models.move_to_end(name)  # LRU の更新
        return model

    if name not in _loaders:
        raise KeyError(f"未登録のモデルです: {name}")

    with _load_lock:
        # ロック待ちの間に他スレッドがロード済みかもしれない
        model = _models.get(name)
        if model is not None:
            _models.move_to_end(name)
            return model

        _evict_if_needed(reserve=1)
        logger.info("loading model: %s", name)
        model = _loaders[name]()
        _models[name] = model
        logger.info("loaded model: %s (resident=%s)", name, list(_models))
        return model


def _evict_if_needed(reserve: int = 0) -> None:
    """常駐上限を超えるモデルを古い順に解放する."""
    limit = get_settings().MAX_RESIDENT_MODELS
    while len(_models) + reserve > limit and _models:
        name, model = _models.popitem(last=False)
        logger.info("unloading model: %s", name)
        _release(model)


def unload(name: str) -> bool:
    """指定モデルを明示的に解放する."""
    model = _models.pop(name, None)
    if model is None:
        return False
    _release(model)
    return True


def unload_all() -> None:
    while _models:
        _, model = _models.popitem(last=False)
        _release(model)


def _release(model: Any) -> None:
    """モデルを解放して VRAM を返す."""
    try:
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception:  # noqa: BLE001
        logger.warning("failed to move model to cpu", exc_info=True)
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@contextmanager
def use_gpu(name: str) -> Iterator[Any]:
    """GPU を専有してモデルを使う.

        with use_gpu("grounding_dino") as model:
            boxes = model.predict(...)

    ロックを取ってからロードするので、ロード中に別リクエストが
    割り込んで VRAM を奪うことがない。
    """
    with gpu_lock:
        yield get_model(name)
