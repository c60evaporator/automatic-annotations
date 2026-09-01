"""ロギング設定.

webapp 側とほぼ同じだが、uvicorn 向けに2点だけ違いがある:

1. **uvicorn のロガーからハンドラを外す。**
   uvicorn は起動時に uvicorn / uvicorn.error / uvicorn.access へ
   独自のハンドラを付ける。こちらでルートにハンドラを足すと、
   同じ行が2回出力される。ハンドラを空にして propagate させ、
   ルート側のフォーマットに一本化する。

2. **スレッド名をフォーマットに含める。**
   推論ジョブは ThreadPoolExecutor のワーカーで走るため、
   どのジョブのログかを追うのにスレッド名が要る。
"""
import logging
import sys
from logging.config import dictConfig

from app.core.config import get_settings

_configured = False

# uvicorn が独自にハンドラを付けるロガー
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def setup_logging(force: bool = False) -> None:
    """ロギングを初期化する.

    uvicorn --reload はプロセスを作り直すが、モジュールの再 import で
    多重に初期化されうるのでフラグでガードする。
    """
    global _configured
    if _configured and not force:
        return

    level = get_settings().LOG_LEVEL.upper()

    dictConfig({
        "version": 1,
        # uvicorn が先に作ったロガーを消さない（消すと起動ログが出なくなる）
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": (
                    "%(asctime)s %(levelname)-8s [%(name)s] "
                    "(%(threadName)s) %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "format": "%(asctime)s %(levelname)-8s [access] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
            },
            "access": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "access",
            },
        },
        "root": {"handlers": ["console"], "level": level},
        "loggers": {
            # ハンドラを空にして propagate させ、二重出力を防ぐ
            "uvicorn":       {"handlers": [], "level": level, "propagate": True},
            "uvicorn.error": {"handlers": [], "level": level, "propagate": True},
            # アクセスログだけは別フォーマット（propagate は切る）
            "uvicorn.access": {
                "handlers": ["access"], "level": level, "propagate": False,
            },
            # モデル系ライブラリは既定で冗長なので抑える
            "transformers":  {"level": "WARNING", "propagate": True},
            "urllib3":       {"level": "WARNING", "propagate": True},
            "httpx":         {"level": "WARNING", "propagate": True},
            "PIL":           {"level": "WARNING", "propagate": True},
            "matplotlib":    {"level": "WARNING", "propagate": True},
        },
    })

    # dictConfig は既存ロガーの handlers を置き換えるが、
    # uvicorn が後から add_handler した場合に備えて明示的に外す
    for name in _UVICORN_LOGGERS:
        if name == "uvicorn.access":
            continue
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
