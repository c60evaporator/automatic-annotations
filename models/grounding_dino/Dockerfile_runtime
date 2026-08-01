# syntax=docker/dockerfile:1
############################
# Build stage for GroundingDINO environment
############################
# Base image for building the GroundingDINO environment
FROM jidohub/base-cu124-devel-ubuntu22.04:latest AS builder

# Freeze the installed packages before installing GroundingDINO dependencies
RUN python -m pip freeze --exclude-editable \
    | LC_ALL=C sort \
    > /tmp/packages-before.txt

# Set environment variables for GroundingDINO build (TORCH_CUDA_ARCH_LIST)
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Clone GroundingDINO repository and install the required dependencies
WORKDIR /workspace
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /workspace/GroundingDINO
# GroundingDINO imports torch from setup.py while resolving build requirements.
# Reuse the environment where torch was installed instead of an isolated build.
RUN pip install \
    --no-build-isolation \
    -e .
# GroundingDINO expects transformers 4.x, so downgrade it
RUN pip install "transformers<5"

# Save the diff of installed packages to a requirements file for runtime environment
RUN mkdir -p /artifacts \
    && python -m pip freeze --exclude-editable \
        | LC_ALL=C sort \
        > /tmp/packages-after.txt \
    && comm -13 \
        /tmp/packages-before.txt \
        /tmp/packages-after.txt \
        > /artifacts/requirements-runtime.txt \
    && echo "Runtime dependencies:" \
    && cat /artifacts/requirements-runtime.txt
# Create an artifact directory and copy the GroundingDINO into it, excluding unnecessary files
RUN cp -a /workspace/GroundingDINO /artifacts/GroundingDINO && \
    rm -rf /workspace/GroundingDINO/build && \
    rm -rf /workspace/GroundingDINO/.git && \
    find /workspace/GroundingDINO \
        -name "__pycache__" -type d -exec rm -rf {} +

############################
# Runtime stage for GroundingDINO environment
############################
# Base image for GroundingDINO runtime environment
FROM jidohub/base-cu124-runtime-ubuntu22.04:latest

# Set environment variables for GroundingDINO runtime (CUDA_HOME)
ENV CUDA_HOME=/usr/local/cuda
# GroundingDINO Python API needs http proxy to download
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

WORKDIR /workspace

# Install the runtime dependencies defined in the requirements-runtime.txt file generated in the builder stage
COPY --from=builder \
    /artifacts/requirements-runtime.txt \
    /tmp/requirements-runtime.txt
RUN python -m pip install \
        --no-cache-dir \
        -r /tmp/requirements-runtime.txt \
    && rm -f /tmp/requirements-runtime.txt
# Copy the GroundingDINO environment from the builder stage
COPY --from=builder \
    /artifacts/GroundingDINO \
    /workspace/GroundingDINO
ENV PYTHONPATH=/workspace/GroundingDINO:${PYTHONPATH}

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
