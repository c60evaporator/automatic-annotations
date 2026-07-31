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

Build the container

```bash
cd ..
docker compose build
```

Run the container

```bash
docker compose up
```

