# syntax=docker/dockerfile:1

############################
# Set the base image with CUDA and cuDNN
############################
# CUDA and Ubuntu Versions
ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

# Base image
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

############################
# Install Python, PyTorch, and other dependencies
############################
# Python version
ARG PYTHON_VERSION=3.11
# PyTorch versions
ARG TORCH_CUDA=cu124
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
# Install Python and other apt packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git curl wget ca-certificates \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip
# pip upgrade
RUN pip install --upgrade pip setuptools wheel
# Install PyTorch and torchvision
RUN pip install \
      torch==${TORCH_VERSION}+${TORCH_CUDA} \
      torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA} \
      --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

############################
# Install GroundingDINO
############################
# Set environment variables for GroundingDINO (TORCH_CUDA_ARCH_LIST and CUDA_HOME)
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CUDA_HOME=/usr/local/cuda
# GroundingDINO Python API needs http proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
# Clone GroundingDINO repository and install the required dependencies
WORKDIR /workspace
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /workspace/GroundingDINO
# GroundingDINO imports torch from setup.py while resolving build requirements.
# Reuse the environment where torch was installed instead of an isolated build.
RUN python -m pip install --no-build-isolation -e .
# GroundingDINO expects transformers 4.x, so downgrade it
RUN python -m pip install "transformers<5"
WORKDIR /workspace

# Install additional Python packages
RUN pip install jupyter seaborn

############################
# Create non-root user
############################
ARG USERNAME=developer
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && chown -R ${UID}:${GID} /workspace
USER ${USERNAME}
