#!/usr/bin/env bash
# Apple Silicon Mac 上で推論サーバーをネイティブに起動する（MPS 推論）.
#
# 環境は ./scripts/setup_mac.sh で作っておくこと。
# webapp は docker compose -f docker-compose.mac.yml up でコンテナとして起動し、
# host.docker.internal:8000 経由でこのサーバーへ接続する。
#
# パスは webapp/.env から読む（docker-compose.mac.yml と同じ値を使うため）:
#   HOST_DATA_ROOT     データセットのルート（必須）
#   HOST_DERIVED_ROOT  深度マップ・点群の保存先（既定: webapp/derived）

set -euo pipefail

INFERENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBAPP_DIR="$(cd "${INFERENCE_DIR}/.." && pwd)"
MAC_DIR="${INFERENCE_DIR}/.mac"
VENV_DIR="${MAC_DIR}/venv"

if [[ ! -x "${VENV_DIR}/bin/uvicorn" ]]; then
    echo "Native environment not found. Run ${INFERENCE_DIR}/scripts/setup_mac.sh first." >&2
    exit 1
fi

# Load only the needed keys from webapp/.env
# (sourcing the whole file fails because UID is a readonly variable in bash)
read_env() {
    local key="$1"
    [[ -n "${!key:-}" || ! -f "${WEBAPP_DIR}/.env" ]] && return 0
    local line
    line="$(grep -E "^[[:space:]]*${key}=" "${WEBAPP_DIR}/.env" | tail -n 1)" || return 0
    line="${line#*=}"
    line="${line%%#*}"
    line="$(echo "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/")"
    [[ -n "${line}" ]] && export "${key}=${line}"
    return 0
}
for key in HOST_DATA_ROOT HOST_DERIVED_ROOT HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    read_env "${key}"
done
if [[ -n "${HTTP_PROXY:-}" ]]; then export http_proxy="${HTTP_PROXY}"; fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then export https_proxy="${HTTPS_PROXY}"; fi
: "${HOST_DATA_ROOT:?HOST_DATA_ROOT is not set in ${WEBAPP_DIR}/.env}"

# Relative paths are resolved against webapp/ (same as docker compose)
resolve_path() {
    case "$1" in
        /*) echo "$1" ;;
        *)  echo "${WEBAPP_DIR}/${1#./}" ;;
    esac
}
HOST_DERIVED_ROOT="$(resolve_path "${HOST_DERIVED_ROOT:-./derived}")"
mkdir -p "${HOST_DERIVED_ROOT}"

# The common package is mounted to /opt/shared/common in the containers.
# Expose only webapp/common (not the whole webapp/ directory) via a symlink
SHARED_DIR="${MAC_DIR}/shared"
mkdir -p "${SHARED_DIR}"
ln -sfn "${WEBAPP_DIR}/common" "${SHARED_DIR}/common"

export DATA_ROOT="$(resolve_path "${HOST_DATA_ROOT}")"
export DERIVED_ROOT="${HOST_DERIVED_ROOT}"
export PYTHONPATH="${INFERENCE_DIR}:${SHARED_DIR}"
export PYTHONUNBUFFERED=1
# Fall back to CPU for operators not implemented on MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1
export NO_PROXY="localhost,127.0.0.1,host.docker.internal${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

# Docker Desktop forwards host.docker.internal to the host's loopback, so 127.0.0.1 is enough
INFERENCE_HOST="${INFERENCE_HOST:-127.0.0.1}"
INFERENCE_PORT="${INFERENCE_PORT:-8000}"

cd "${INFERENCE_DIR}"
# --workers should be always one to prevent multiple model load (memory overflow)
# --reload-dir limits the watch to app/ (.mac/ contains the venv)
exec "${VENV_DIR}/bin/uvicorn" app.main:app \
    --host "${INFERENCE_HOST}" --port "${INFERENCE_PORT}" \
    --workers 1 --reload --reload-dir app
