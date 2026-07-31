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

Build the container

```bash
cd ..
docker compose build
```

Run the container

```bash
docker compose up
```
