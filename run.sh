#!/bin/bash

# run profile (including validation)

# Llama3.1-8B
python3 run_profile.py \
  --hardware RTX3090 --model "meta-llama/Llama-3.1-8B" --num-layers 1 --device cuda \
  --warmup 10 --repeat 30 --mode "both" --prefill-max 2048 --decode-max 2048 \
  --scaling-num-layers 32 --scaling-input-min 128 --scaling-input-max 1024 --scaling-input-step 128 --scaling-repeat 3 \
  --hf-token "<your_token>" --verbose

# Phi-mini-MoE
python3 run_profile.py \
  --hardware RTX3090 --model "microsoft/Phi-mini-MoE-instruct" --num-layers 1 --device cuda \
  --warmup 10 --repeat 30 --mode "both" --prefill-max 2048 --decode-max 2048 \
  --scaling-num-layers 32 --scaling-input-min 128 --scaling-input-max 1024 --scaling-input-step 128 --scaling-repeat 3 \
  --hf-token "<your_token>" --verbose