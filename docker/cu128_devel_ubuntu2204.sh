#!/usr/bin/env bash

set -euo pipefail

# Configuration
CUDA_VERSION="${CUDA_VERSION:-12.8.0}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"

# Build
IMAGE_TAG="jidohub/base-${TORCH_CUDA}-devel-ubuntu${UBUNTU_VERSION}:latest"
DOCKERFILE="Dockerfile_devel"
BUILD_CONTEXT="."

docker build \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
    --build-arg TORCH_CUDA="${TORCH_CUDA}" \
    --build-arg TORCH_VERSION="${TORCH_VERSION}" \
    --build-arg TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_TAG}" \
    "${BUILD_CONTEXT}"
