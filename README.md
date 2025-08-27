# LLM-Profile
LLM-Profile is a profiling tool designed to support GPU execution within [LLMServingSim](https://github.com/casys-kaist/LLMServingSim). 

## Overview
This repository loads LLMs from Hugging Face and inserts PyTorch profiler hooks into key layers to measure their execution time on GPU. 

It is intended for performance analysis and simulation purposes, especially in conjunction with LLMServingSim.

## How to Use
### 1. Environment Setting
Run the appropriate PyTorch Docker container based on your GPU and CUDA version.
See `docker.sh` for an example.

Alternatively, if you already have PyTorch and CUDA installed natively, Docker is not required.

### 2. Set Hugging Face Token
For models that require access approval (e.g., LLaMA), you must provide your Hugging Face token.

- Add argument `--hf-token` while running `run_profile.py`
```bash
python3 run_profile.py ... --hf-token "<your_token>"
```
- Or, set the token via an environment variable before running the script
```bash
export HF_TOKEN=<your_token>
```

### 3. Run the Profiler
Refer to the `run_profile` function in `run_profile.py` to start profiling. 

If you're working with a large model, you can reduce `num_layers` in the configuration to ensure the model fits within a single GPU.

Once the configuration is determined, simply run `run_profile.py` with arguments. Or, modify `run.sh`.

**Validation is included**, options starting with `--scaling` is for validation. Refer to [here](#4-validation).

```bash
python3 run_profile.py \
  --hardware RTX3090 --model "meta-llama/Llama-3.1-8B" --num-layers 1 --device cuda \
  --warmup 10 --repeat 100 --mode "both" --prefill-max 2048 --decode-max 2048 \
  --scaling-num-layers 32 --scaling-input-min 128 --scaling-input-max 1024 --scaling-input-step 128 --scaling-repeat 3 \
  --hf-token "<your_token>" --verbose
```
The profiling results will be saved in a CSV file named `<hardware>/<model_name>.csv`.

An example profiling result for RTX 3090 is provided in `RTX3090` folder.


### 4. Validation

The `validation.py` module provides functionality to automatically compute and apply a scaling factor to adjust the estimated latency.

This step is necessary because measuring only CUDA time does not account for various real-world overheads, such as CPU overhead. The validation process compensates for these discrepancies.


- `compute_average_scaling_factor()` estimates the scaling factor based on the difference between real-model latency and estimated-model latency.

- `apply_scaling_to_latency_csv()` adjusts the latency values in the CSV using the computed scaling factor.

For implementation details, see `validation.py`.