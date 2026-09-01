# syntax=docker/dockerfile:1
# inference/Dockerfile_devel

# Base image
FROM jidohub/base-cu124-devel-ubuntu22.04:latest

# Common environment variables (TORCH_CUDA_ARCH_LIST)
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# プロキシはビルド時のみ有効にする（ENV にすると実行時にも残り、
# プロキシの無い環境で外部アクセスが全て失敗するようになる）
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy

# サードパーティのリポジトリは /opt に置く。
# /workspace は docker-compose で ./inference をバインドマウントするため、
# ここに clone するとマウント時に丸ごと隠れて `pip install -e` の
# 参照先が消え、import が失敗する。
ENV THIRD_PARTY_ROOT=/opt/third_party
RUN mkdir -p ${THIRD_PARTY_ROOT}

# モデル重みのキャッシュ先。ボリュームでマウントして再ダウンロードを防ぐ
ENV HF_HOME=/opt/cache/huggingface \
    TORCH_HOME=/opt/cache/torch \
    CHECKPOINT_DIR=/opt/checkpoints
RUN mkdir -p ${HF_HOME} ${TORCH_HOME} ${CHECKPOINT_DIR}

############################
# SAM2 Installation
############################
WORKDIR ${THIRD_PARTY_ROOT}
RUN git clone https://github.com/facebookresearch/sam2.git
WORKDIR ${THIRD_PARTY_ROOT}/sam2
# SAM2 uses CUDAExtension build that needs the same torch environment.
# Reuse the environment where torch was installed instead of an isolated build.
RUN pip install \
    --no-build-isolation \
    -e .

############################
# Grounding DINO Installation
############################
# Environment variable for GroundingDINO (CUDA_HOME)
ENV CUDA_HOME=/usr/local/cuda
WORKDIR ${THIRD_PARTY_ROOT}
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR ${THIRD_PARTY_ROOT}/GroundingDINO
# GroundingDINO imports torch from setup.py while resolving build requirements.
# Reuse the environment where torch was installed instead of an isolated build.
RUN pip install \
    --no-build-isolation \
    -e .
# GroundingDINO expects transformers 4.x, so downgrade it
RUN pip install "transformers<5"

############################
# Depth-Anything-3 Installation
############################
RUN pip install xformers==0.0.28.post3
WORKDIR ${THIRD_PARTY_ROOT}
RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
WORKDIR ${THIRD_PARTY_ROOT}/Depth-Anything-3
# Depth-Anything-3 doesn't need torch for building, so `--no-build-isolation` argument is not used
RUN pip install -e .
# Install Gaussian head
RUN pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

############################
# Other dependencies
############################
# Patchwork++
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeigen3-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*
RUN pip install pypatchworkpp

############################
# API server dependencies
############################
# NOTE: `pip install fastapi` だけでは uvicorn は入らない
#       （fastapi の依存は starlette / pydantic などのみ）。
#       CMD の `--reload` には watchfiles も必要なので standard を入れる。
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

############################
# Import smoke test
############################
# ここで落とすことで、実行時（最初の推論リクエスト）ではなくビルド時に
# 依存の壊れを検出する。
# 特に numpy: ベースイメージの C 拡張は numpy 1.x でビルドされているため、
# 追加インストールで 2.x へ上がると matplotlib などが
# "compiled using NumPy 1.x cannot be run in NumPy 2.x" で落ちる。
# GroundingDINO は visualizer 経由で matplotlib を import するので直撃する。
RUN python -c "\
import numpy, matplotlib, torch, torchvision; \
assert numpy.__version__.startswith('1.'), f'numpy が 1.x ではありません: {numpy.__version__}'; \
import matplotlib.pyplot; \
import groundingdino.util.inference; \
import sam2; \
print('smoke test OK: numpy', numpy.__version__, '/ torch', torch.__version__)"

############################
# Create non-root user
############################
ARG USERNAME=developer
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# アプリのコード置き場。マウントされるので中身は空でよいが、
# 所有者を先に合わせておく
RUN mkdir -p /workspace /derived \
    && chown -R ${UID}:${GID} /workspace /derived ${THIRD_PARTY_ROOT} /opt/cache ${CHECKPOINT_DIR}

USER ${USERNAME}
WORKDIR /workspace

# uvicorn は CWD を sys.path に入れる（--app-dir の既定値が "."）が、
# 明示しておくとサブプロセスからの import でも確実になる
ENV PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1

EXPOSE 8000

############################
# Run FastAPI
############################
# --workers は必ず 1。GPU モデルをプロセス数だけロードすることになり
# VRAM が溢れる
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--reload"]
