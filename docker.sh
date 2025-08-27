#!/bin/bash

# launch pytorch docker
docker run --name llm_profile \
  --gpus '"device=0"' \
  -it \
  -v $(pwd):/workspace \
  --shm-size=16g \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -lc 'set -euo pipefail
  apt-get update && apt-get install -y git ninja-build cmake && \
  pip install -U pip setuptools wheel packaging && \
  pip install -U transformers && \
  exec bash'