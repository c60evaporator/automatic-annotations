## Build the base Docker image

### cu124-devel-ubuntu22.04

devel

```
cd docker
docker build -t jidohub/base-cu124-devel-ubuntu22.04:latest -f Dockerfile_devel .
```

runtime

```
cd docker
docker build -t jidohub/base-cu124-runtime-ubuntu22.04:latest -f Dockerfile_runtime .
```
