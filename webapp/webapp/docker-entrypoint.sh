#!/usr/bin/env bash
# webapp/docker-entrypoint.sh
#
# 起動時に必ずマイグレーションを適用してから CMD を実行する。
# CLI（インポート等）を差し替え実行する場合も同じく適用される。
set -euo pipefail

cd /workspace

if [ "${SKIP_MIGRATION:-0}" != "1" ]; then
  echo "[entrypoint] applying migrations..."
  alembic upgrade head
  echo "[entrypoint] migrations up to date."
fi

exec "$@"
