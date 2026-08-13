## Installation

Clone the weight of sam2.1_hiera_large

```bash
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

If you want to use [other checkpoints](https://github.com/facebookresearch/sam2) like sam2.1_hiera_base_plus, please download its weight as well

```bash
wget wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

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
