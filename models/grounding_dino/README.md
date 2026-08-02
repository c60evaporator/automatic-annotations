## Installation

Clone the weight of GroundingDINO-B

```bash
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

If you want to use GroundingDINO-T, please download its weight as well

```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

export `TORCH_CUDA_ARCH_LIST` environment variable based on [this page](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) 

```
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
