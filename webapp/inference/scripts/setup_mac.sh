#!/usr/bin/env bash
# Apple Silicon Mac 上に推論サーバーのネイティブ環境（MPS 対応）を作る.
#
# Docker on macOS からは Metal（MPS）が使えないため、推論サーバーだけをホストで動かす。
# インストール内容は Dockerfile_devel に合わせている（差分は xformers を入れず、pycolmap をスタブにすることのみ）。
# 作成先: webapp/inference/.mac/{venv,third_party}（.gitignore 済み）
#
# 使い方: ./scripts/setup_mac.sh   （起動は ./scripts/run_mac.sh）
# 前提:   uv と libusb（brew install uv libusb）。Python 3.11 は uv が自動で取得する

set -euo pipefail

# Configuration (aligned with docker/cpu_devel_ubuntu2204.sh and Dockerfile_devel)
PYTHON_VERSION="${PYTHON_VERSION:-3.11.11}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.1}"
DA3_COMMIT="${DA3_COMMIT:-3d835ec1a5802d64a8b8b15f817a1ab54809bfe4}"
MAPANYTHING_VERSION="${MAPANYTHING_VERSION:-v1.1.3}"
PARTHOE_COMMIT="${PARTHOE_COMMIT:-6848ad2477dad39c2c1caf3d1fc3d7a5f1970363}"

INFERENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${INFERENCE_DIR}/../.." && pwd)"
MAC_DIR="${INFERENCE_DIR}/.mac"
VENV_DIR="${MAC_DIR}/venv"
THIRD_PARTY_ROOT="${MAC_DIR}/third_party"
CONSTRAINTS="${MAC_DIR}/pip-constraints.txt"
PYTHON="${VENV_DIR}/bin/python"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This script is for Apple Silicon Macs (Darwin arm64)." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: brew install uv" >&2
    exit 1
fi
# The open3d macOS wheel links /opt/homebrew/opt/libusb (libusb-1.0-0 in Dockerfile_devel)
if [[ ! -e /opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib ]]; then
    echo "libusb is required by open3d: brew install libusb" >&2
    exit 1
fi

pip_install() {
    uv pip install --python "${PYTHON}" "$@"
}

############################
# Python and PyTorch
############################
mkdir -p "${MAC_DIR}" "${THIRD_PARTY_ROOT}"
if [[ ! -x "${PYTHON}" ]]; then
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi
# PyPI の macOS arm64 wheel は MPS 対応でビルドされている
pip_install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
# Same libraries as the base image
pip_install \
    -r "${REPO_ROOT}/docker/constraints_cpu_runtime.txt" \
    -r "${REPO_ROOT}/docker/constraints_cpu_devel.txt"

############################
# Common constraints
############################
# Pin torch/torchvision/torchaudio/numpy so that the dependencies cannot replace them
# (pyarrow>=26 doesn't support numpy 1.x, opencv-python is aligned with opencv-python-headless of MapAnything)
uv pip freeze --python "${PYTHON}" | grep -iE "^(torch|torchvision|torchaudio|numpy)==" > "${CONSTRAINTS}"
echo "pyarrow<26" >> "${CONSTRAINTS}"
echo "opencv-python==4.10.0.84" >> "${CONSTRAINTS}"

############################
# GroundingDINO / SAM2 / SigLIP2 (Transformers)
############################
pip_install -c "${CONSTRAINTS}" "transformers~=5.20.0"

############################
# Depth-Anything-3
############################
DA3_DIR="${THIRD_PARTY_ROOT}/Depth-Anything-3"
if [[ ! -d "${DA3_DIR}" ]]; then
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "${DA3_DIR}"
fi
git -C "${DA3_DIR}" fetch --quiet origin
git -C "${DA3_DIR}" checkout --quiet "${DA3_COMMIT}"
# The following dependencies in pyproject.toml are excluded (addict is imported but not listed in the dependencies):
#   - xformers: CUDA only. It is used optionally (SwiGLU falls back to PyTorch)
#   - pycolmap: its wheel bundles another libomp, which aborts the process together with torch's libomp
#               (OMP: Error #15). It is only used by the COLMAP export but imported eagerly by
#               depth_anything_3.api, so the stub module (same as Dockerfile_cpu) is installed instead
"${PYTHON}" -c "import tomllib, sys; deps = tomllib.load(open(sys.argv[1], 'rb'))['project']['dependencies']; print('\n'.join(d for d in deps if not d.startswith(('xformers', 'pycolmap'))))" \
    "${DA3_DIR}/pyproject.toml" > "${MAC_DIR}/requirements_da3.txt"
pip_install -c "${CONSTRAINTS}" -r "${MAC_DIR}/requirements_da3.txt" addict
pip_install -c "${CONSTRAINTS}" --no-deps -e "${DA3_DIR}"
uv pip uninstall --python "${PYTHON}" pycolmap 2>/dev/null || true
cp "${INFERENCE_DIR}/stubs/pycolmap.py" \
    "$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/pycolmap.py"

############################
# MapAnything
############################
MAPANYTHING_DIR="${THIRD_PARTY_ROOT}/map-anything"
if [[ ! -d "${MAPANYTHING_DIR}" ]]; then
    git clone --branch "${MAPANYTHING_VERSION}" --depth 1 https://github.com/facebookresearch/map-anything.git "${MAPANYTHING_DIR}"
fi
# Install the dependencies except rerun-sdk~=0.24.1 and plyfile>=1.1.4 (they require numpy>=2)
"${PYTHON}" -c "import tomllib, sys; deps = tomllib.load(open(sys.argv[1], 'rb'))['project']['dependencies']; print('\n'.join(d for d in deps if not d.startswith('rerun-sdk')))" \
    "${MAPANYTHING_DIR}/pyproject.toml" > "${MAC_DIR}/requirements_mapanything.txt"
pip_install -c "${CONSTRAINTS}" -r "${MAC_DIR}/requirements_mapanything.txt" "plyfile<1.1.4"
pip_install -c "${CONSTRAINTS}" --no-deps -e "${MAPANYTHING_DIR}"
# Pre-fetch the DINOv2 code loaded via torch hub (into TORCH_HOME, default ~/.cache/torch)
"${PYTHON}" -c "import torch; torch.hub.list('facebookresearch/dinov2', trust_repo=True)"

############################
# PartHOE
############################
# Only MinimalPartHOE in tensorrt/nn_utils.py is used (torch only, no extra dependencies)
PARTHOE_DIR="${THIRD_PARTY_ROOT}/Part_HOE"
if [[ ! -d "${PARTHOE_DIR}" ]]; then
    git clone https://github.com/zhaojieting/Part_HOE.git "${PARTHOE_DIR}"
fi
git -C "${PARTHOE_DIR}" fetch --quiet origin
git -C "${PARTHOE_DIR}" checkout --quiet "${PARTHOE_COMMIT}"

############################
# Other dependencies
############################
# Patchwork++ (ground removal) and open3d (point cloud processing)
pip_install -c "${CONSTRAINTS}" pypatchworkpp open3d

############################
# API server dependencies
############################
pip_install -c "${CONSTRAINTS}" -r "${INFERENCE_DIR}/requirements.txt"

############################
# Import smoke test
############################
"${PYTHON}" -c "\
import numpy, matplotlib, torch, torchvision; \
assert numpy.__version__.startswith('1.'), f'numpy が 1.x ではありません: {numpy.__version__}'; \
import matplotlib.pyplot; \
from transformers import GroundingDinoForObjectDetection, Sam2Model, Sam2VideoModel, Siglip2Model; \
import depth_anything_3.api; \
import mapanything.models; \
import open3d, pypatchworkpp, cv2, fastapi, shapely; \
import importlib.util; \
spec = importlib.util.spec_from_file_location('parthoe_nn_utils', '${PARTHOE_DIR}/tensorrt/nn_utils.py'); \
spec.loader.exec_module(importlib.util.module_from_spec(spec)); \
assert torch.backends.mps.is_available(), 'MPS が使えません'; \
torch.randn(256, 256) @ torch.randn(256, 256); \
print('smoke test OK: numpy', numpy.__version__, '/ torch', torch.__version__, '/ mps available')"

echo "Done. Start the inference server with: ${INFERENCE_DIR}/scripts/run_mac.sh"
