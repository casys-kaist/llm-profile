import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
from patch_model import *
from validation import *
from collections import defaultdict
from tqdm import tqdm
import csv
import os
import gc
import argparse

def run_profile(
    hardware="RTX3090",
    model_name="meta-llama/Llama-3.1-8B",
    num_layers=None,
    input_lengths=[128, 256, 512, 1024],
    kv_cache_lengths=[0, 128, 512, 1024],
    device="cuda",
    warmup=10,
    repeat=100,
    csv_append=True,
    verbose=False
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()

    # Reduce model layers if specified
    if num_layers is not None:
        if 'llama' in model_name.lower() or 'phi' in model_name.lower():
            model.model.layers = model.model.layers[:num_layers]
        else:
            model.model.decoder.layers = model.model.decoder.layers[:num_layers]
        model.config.num_hidden_layers = num_layers

    model.to(device)

    patch_model(model, model.config)

    results = defaultdict(float)
    total_tasks = len(input_lengths) * len(kv_cache_lengths)
    csv_rows = []

    for (input_len, kv_len) in tqdm([(l, k) for l in input_lengths for k in kv_cache_lengths], total=total_tasks, desc="Profiling configs"):
        if input_len + kv_len > model.config.max_position_embeddings:
            continue  # Skip if input length exceeds max position embeddings
        if verbose:
            print(f"\nRunning input_len={input_len}, kv_len={kv_len}",flush=True)

        input_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(1, input_len), device=device)

        num_layers = model.config.num_hidden_layers
        kv_head = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else model.config.num_attention_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        if 'llama' in model_name.lower():
            past_key_values = create_llama_past_key_values(model.config, kv_len, device)
        elif 'phi' in model_name.lower():
            past_key_values = create_phimoe_past_key_values(model.config, kv_len, device)
        else:
            # OPT-compatible tuple
            past_key_values = tuple([
                torch.zeros((1, kv_head, kv_len, head_dim), device=device),
                torch.zeros((1, kv_head, kv_len, head_dim), device=device)
            ])
        
        # Warm-up phase
        for _ in range(warmup):
            if 'llama' in model_name.lower():
                past_key_values = create_llama_past_key_values(model.config, kv_len, device)
            elif 'phi' in model_name.lower():
                past_key_values = create_phimoe_past_key_values(model.config, kv_len, device)
            with torch.no_grad():
                _ = model(input_ids, past_key_values=past_key_values, use_cache=True)

        # Profiling phase
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
            with_modules=False,
        ) as prof:
            for _ in range(repeat):
                if 'llama' in model_name.lower():
                    past_key_values = create_llama_past_key_values(model.config, kv_len, device)
                elif 'phi' in model_name.lower():
                    past_key_values = create_phimoe_past_key_values(model.config, kv_len, device)
                with torch.no_grad():
                    _ = model(input_ids, past_key_values=past_key_values, use_cache=True)
                prof.step()

        # For experts
        w1 = 0
        w2 = 0
        w3 = 0
        expert_counts = 0
        # probe_profiler(prof, max_events=5)  # For debugging
        for evt in prof.key_averages():
            if evt.key.startswith("aten::"):
                continue
            if any(kw in evt.key for kw in ["embedding", "layernorm", "proj", "fc", "matmul", "softmax", "act_fn", "rope", "gate", "sparsemixer", "attn", "lm_head"]):
                cpu_time = evt.cpu_time
                time_us = evt.device_time
                if cpu_time > 0 and time_us > 0:
                    if verbose:
                        print(f"input={input_len}, kv={kv_len}, layer={evt.key}, time={time_us:.2f} us")
                    key = (input_len, kv_len, evt.key)
                    results[key] = time_us
                    csv_rows.append({
                        "hardware": hardware,
                        "model": model_name,
                        "layer_name": evt.key,
                        "input": input_len,
                        "kv_cache": kv_len,
                        "latency(ns)": int(time_us * 1000)  # convert us to ns
                    })
            
            if "expert" in evt.key:
                cpu_time = evt.cpu_time
                time_us = evt.device_time
                if cpu_time > 0 and time_us > 0:
                    if 'w1' in evt.key:
                        w1 += time_us
                    elif 'w2' in evt.key:
                        w2 += time_us
                    elif 'w3' in evt.key:
                        w3 += time_us
                    expert_counts += 1
        
        if expert_counts > 0:
            if verbose:
                print(f"input={input_len}, kv={kv_len}, layer=expert.w1, time={w1:.2f} us")
                print(f"input={input_len}, kv={kv_len}, layer=expert.w2, time={w2:.2f} us")
                print(f"input={input_len}, kv={kv_len}, layer=expert.w3, time={w3:.2f} us")
            results[(input_len, kv_len, "expert.w1")] = w1
            results[(input_len, kv_len, "expert.w2")] = w2
            results[(input_len, kv_len, "expert.w3")] = w3
            csv_rows.append({
                "hardware": hardware,
                "model": model_name,
                "layer_name": "expert.w1",
                "input": input_len,
                "kv_cache": kv_len,
                "latency(ns)": int(w1 * 1000)  # convert us to ns
            })
            csv_rows.append({
                "hardware": hardware,
                "model": model_name,
                "layer_name": "expert.w2",
                "input": input_len,
                "kv_cache": kv_len,
                "latency(ns)": int(w2 * 1000)  # convert us to ns
            })
            csv_rows.append({
                "hardware": hardware,
                "model": model_name,
                "layer_name": "expert.w3",
                "input": input_len,
                "kv_cache": kv_len,
                "latency(ns)": int(w3 * 1000)  # convert us to ns
            })
                    

        embedding = results.get((input_len, kv_len, "embedding"), 0.0)
        final_norm = results.get((input_len, kv_len, "final_layernorm"), 0.0)
        lm_head = results.get((input_len, kv_len, "lm_head"), 0.0)
        if 'llama' in model_name.lower():
            block_components = [
                "input_layernorm",
                "q_proj",
                "k_proj",
                "v_proj",
                "rope",
                "attn",
                "o_proj",
                "post_layernorm",
                "gate_proj",
                "up_proj",
                "act_fn",
                "down_proj"
            ]
        elif 'phi' in model_name.lower():
            block_components = [
                "input_layernorm",
                "q_proj",
                "k_proj",
                "v_proj",
                "rope",
                "attn",
                "o_proj",
                "post_layernorm",
                "gate",
                "sparsemixer"
            ]            
        else:  # OPT/TransformerDecoder
            block_components = [
                "input_layernorm",
                "q_proj",
                "k_proj",
                "v_proj",
                "qk_matmul",
                "softmax",
                "sv_matmul",
                "o_proj",
                "post_layernorm",
                "fc1",
                "act_fn",
                "fc2"
            ]
        per_block_time = sum(results.get((input_len, kv_len, comp), 0.0) for comp in block_components)
        # Runs experts sequentially in huggungface implementation
        if 'phi' in model_name.lower():
            moe_components = ["expert.w1", "expert.w2", "expert.w3", "act_fn"]
            per_block_time += sum(results.get((input_len, kv_len, comp), 0.0) for comp in moe_components) * model.config.num_local_experts

        full_latency_estimate = embedding + final_norm + lm_head + per_block_time * num_layers

        if verbose:
            print(f"Estimated latency: {(full_latency_estimate / 1000):.2f} ms")

    output_path = f"{hardware}/{model_name}.csv"
    if csv_append:
        mode = "a"
    else:
        mode = "w"
    file_exists = os.path.exists(output_path)
    if not file_exists:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["hardware", "model", "layer_name", "input", "kv_cache", "latency(ns)"])
        if not file_exists or not csv_append:
            writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Writing profiled results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="LLM micro-profiler (prefill/decode + scaling)")
    # Common
    parser.add_argument("--hardware", default="RTX3090")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--append", action="store_true", help="append to CSV instead of overwrite")
    parser.add_argument("--verbose", action="store_true")

    # Which stages to run
    parser.add_argument("--mode", type=str, default="both", choices=["prefill","decode","both"])
    parser.add_argument("--skip-scaling", action="store_true", help="Skip scaling-factor estimation")

    # Prefill sweep (input 1..prefill_max, kv=0)
    parser.add_argument("--prefill-max", type=int, default=2048)

    # Decode sweep (input=1, kv 0..decode_max)
    parser.add_argument("--decode-max", type=int, default=2048)

    # Scaling-factor options (Used to validate the result)
    parser.add_argument("--scaling-num-layers", type=int, default=None, help="Number of layers used in validation. Higher is accurate but can result OOM")
    parser.add_argument("--scaling-input-min", type=int, default=128)
    parser.add_argument("--scaling-input-max", type=int, default=1024)
    parser.add_argument("--scaling-input-step", type=int, default=128)
    parser.add_argument("--scaling-repeat", type=int, default=3)
    parser.add_argument("--scaling-output", type=str, default=None, help="Output path for scaled CSV (default: <original>_scaled.csv)")
    parser.add_argument("--scaling-add-new-col", action="store_true", help="Add extra scaled_latency(ns) column")

    # HF token (optional)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))

    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # ---------- Prefill sweep ----------
    if args.mode in ("prefill","both"):
        run_profile(
            hardware=args.hardware,
            model_name=args.model,
            input_lengths=range(1, args.prefill_max + 1),
            kv_cache_lengths=range(0, 1),
            num_layers=args.num_layers,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            csv_append=False if not args.append else True,
            verbose=args.verbose,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # ---------- Decode sweep ----------
    if args.mode in ("decode","both"):
        run_profile(
            hardware=args.hardware,
            model_name=args.model,
            input_lengths=range(1, 2),
            kv_cache_lengths=range(0, args.decode_max + 1),
            num_layers=args.num_layers,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            csv_append=True,
            verbose=args.verbose,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # ---------- Scaling-factor estimation ----------
    # Reducing the number of layers for profiling often leads to artificially slower per-layer latency due to GPU execution behavior. 
    # This is because GPUs are not fully utilized when the model is shallow. 
    # To compensate for this under-utilization, we measure a scaling factor and apply it to the estimated latency. 
    # As the number of profiled layers approaches the actual model depth, the scaling factor converges to 1.
    if not args.skip_scaling:
        scaling_factor = compute_average_scaling_factor(
            hardware=args.hardware,
            model_name=args.model,
            num_layers=args.scaling_num_layers,
            input_lengths=range(args.scaling_input_min, args.scaling_input_max + 1, args.scaling_input_step),
            output_lengths=range(args.scaling_input_min, args.scaling_input_max + 1, args.scaling_input_step),
            repeat=args.scaling_repeat,
            verbose=True,
        )
        # apply to CSV (in-place overwrite by default)
        apply_scaling_to_latency_csv(
            hardware=args.hardware, model_name=args.model, scaling_factor=scaling_factor, output_path=args.scaling_output, overwrite=not args.scaling_add_new_col
        )


# ---------- Helper functions ----------
def _attrs(obj):
    names = [n for n in dir(obj) if not n.startswith("_")]
    items = []
    for n in names:
        try:
            v = getattr(obj, n)
        except Exception:
            v = "<error>"

        if isinstance(v, (int, float, str, type(None))):
            items.append((n, type(v).__name__, v))
        else:
            items.append((n, type(v).__name__, "..."))
    return items

def _num_time_attrs(attrs):
    keys = ("time", "cpu", "cuda", "count", "calls")
    return [(n, t, v) for (n, t, v) in attrs if any(k in n.lower() for k in keys)]

def probe_profiler(prof, max_events=20):
    print("=== key_averages() ===")
    kav = prof.key_averages()
    for i, e in enumerate(kav[:max_events]):
        print(f"[{i}] key={getattr(e,'key',None)!r}  type={type(e).__name__}")
        attrs = _attrs(e)
        times = _num_time_attrs(attrs)
        for n, t, v in times:
            print(f"   - {n}: {v}")
        print("   -- all attrs:", [n for (n, _, _) in attrs])

    if hasattr(prof, "events"):
        evs = prof.events()
        print(f"\n=== events()  count={len(evs)} ===")
        for i, e in enumerate(evs[:max_events]):
            name = getattr(e, "name", getattr(e, "key", None))
            print(f"[{i}] name={name!r}  type={type(e).__name__}")
            attrs = _attrs(e)
            times = _num_time_attrs(attrs)
            for n, t, v in times:
                print(f"   - {n}: {v}")
            print("   -- all attrs:", [n for (n, _, _) in attrs])
    
if __name__ == "__main__":
    main()