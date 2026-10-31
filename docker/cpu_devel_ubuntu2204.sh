#!/usr/bin/env bash

set -euo pipefail

# Configuration
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"

# Build
IMAGE_TAG="jidohub/base-cpu-devel-ubuntu${UBUNTU_VERSION}:latest"
DOCKERFILE="Dockerfile_cpu_devel"
BUILD_CONTEXT="."

docker build \
    --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
    --build-arg TORCH_VERSION="${TORCH_VERSION}" \
    --build-arg TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_TAG}" \
    "${BUILD_CONTEXT}"
