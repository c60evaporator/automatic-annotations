"""ロギング設定"""
import logging
import sys
from logging.config import dictConfig

from app.core.config import get_settings

_configured = False


def setup_logging(force: bool = False) -> None:
    """ロギングを初期化する.

    Streamlit はスクリプトを繰り返し再実行するため、多重初期化で
    ハンドラが増殖しないようフラグでガードする。
    """
    global _configured
    if _configured and not force:
        return

    level = get_settings().LOG_LEVEL.upper()
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
            },
        },
        "root": {"handlers": ["console"], "level": level},
        "loggers": {
            # SQL ログは SQL_ECHO で別途制御する
            "sqlalchemy.engine": {"level": "WARNING", "propagate": True},
            "alembic": {"level": "INFO", "propagate": True},
        },
    })
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
