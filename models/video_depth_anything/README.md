## Installation

Clone the weight of Metric-Video-Depth-Anything-Large

```bash
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth
```

If you want to use other weights, see [here](https://github.com/DepthAnything/Video-Depth-Anything#pre-trained-models)

export `TORCH_CUDA_ARCH_LIST` environment variable based on [this page](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) 

```bash
export TORCH_CUDA_ARCH_LIST=8.9
```

Build the dev container

```bash
cd ..
export IMAGE_RUNTIME="devel"
docker compose build
```

Build the runtime container

```bash
cd ..
export IMAGE_RUNTIME="runtime"
docker compose build
```

## Usage

If you want to run the devel container, use the following command

```bash
export IMAGE_RUNTIME="devel"
docker compose up
```

If you want to run the runtime container (smaller, but the functions are limited), use the following command

```bash
export IMAGE_RUNTIME="runtime"
docker compose up
```
