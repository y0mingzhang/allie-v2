#!/usr/bin/env python3
"""Scaling law experiments: IsoFLOP curves for chess data.

For each compute budget, train multiple model sizes and measure final val loss + move accuracy.
This reveals the optimal tokens/params ratio for chess data.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import copy

BASE_CONFIG = {
    "distributed": {
        "tp_size": 1,
        "cp_size": 1,
        "pp_size": 1,
        "dp_size": 4,
        "backend": "nccl",
        "use_cpu": False,
    },
    "model": {
        "name": "Qwen/Qwen3-1.7B",
        "use_flash_attention": False,
        "use_fused_adam": True,
        "vocab_size": 2350,
        "max_position_embeddings": 1024,
        "random_init": True,
    },
    "training": {
        "seed": 42,
        "seq_length": 1024,
        "gradient_accumulation_steps": 1,
        "max_tokens": None,
        "grad_clip_norm": 1.0,
        "torch_compile": True,
        "optimizer": {
            "name": "muon",
            "weight_decay": 0.05,
            "momentum": 0.9,
            "ns_coefficients": [3.4445, -4.775, 2.0315],
            "ns_steps": 5,
            "adjust_lr_fn": "match_rms_adamw",
            "muon_eps": 1e-7,
        },
    },
    "dataset": {
        "name": "tokens",
        "train_glob": "data/tokens_v2/full_v1/train.npy",
        "val_glob": "data/tokens_v2/full_v1/val.npy",
        "num_workers": 4,
    },
    "checkpoint": {"save_dir": "", "save_frequency": 999999, "load_path": ""},
    "logging": {"use_wandb": False, "project_name": "chess-v3-scaling", "run_name": ""},
    "environment": {"OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "FLASH_ATTEN": "0"},
}

# Model size configs: (hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads, head_dim)
MODEL_SIZES = {
    "5M": (256, 768, 6, 4, 2, 64),
    "15M": (384, 1152, 8, 6, 3, 64),
    "40M": (512, 1536, 12, 8, 4, 64),
    "100M": (768, 2304, 12, 8, 4, 96),
    "250M": (1024, 3072, 16, 8, 4, 128),
    "600M": (1536, 4608, 16, 12, 4, 128),
}

# IsoFLOP experiments: (compute_label, model_size, total_tokens)
# Designed so each run takes 5-45 min
MAX_DATA = 2.13e9

EXPERIMENTS = []

# Budget 1: 1e17 FLOPs (~small)
for size, tokens in [("5M", 2.13e9), ("15M", 1.1e9), ("40M", 0.42e9), ("100M", 0.17e9)]:
    EXPERIMENTS.append(("1e17", size, min(tokens, MAX_DATA)))

# Budget 2: 3e17 FLOPs (~medium)
for size, tokens in [("15M", 2.13e9), ("40M", 1.3e9), ("100M", 0.53e9), ("250M", 0.2e9)]:
    EXPERIMENTS.append(("3e17", size, min(tokens, MAX_DATA)))

# Budget 3: 1e18 FLOPs (~large)
for size, tokens in [("40M", 2.13e9), ("100M", 1.67e9), ("250M", 0.67e9), ("600M", 0.28e9)]:
    EXPERIMENTS.append(("1e18", size, min(tokens, MAX_DATA)))


def make_config(name, model_size, total_tokens):
    hidden, intermediate, layers, heads, kv_heads, head_dim = MODEL_SIZES[model_size]
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["model"].update(
        {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
        }
    )

    # Batch size: smaller models get bigger batches
    params_M = int(model_size.rstrip("M"))
    if params_M <= 100:
        micro_batch, grad_accum = 64, 1
    elif params_M <= 300:
        micro_batch, grad_accum = 32, 1
    else:
        micro_batch, grad_accum = 16, 2

    batch_tokens = micro_batch * grad_accum * 8 * 1024
    total_steps = max(100, int(total_tokens / batch_tokens))

    # LR: scale with model size. Derisks found higher LR helps for small models.
    # Use sqrt scaling: LR ~ 0.015 * sqrt(153M / N)
    base_lr = 0.015
    lr = base_lr * math.sqrt(153e6 / (params_M * 1e6))
    lr = min(lr, 0.03)  # cap

    warmup = min(50, total_steps // 10)
    decay = max(total_steps // 10, 50)
    stable = total_steps - warmup - decay

    cfg["training"].update(
        {
            "total_train_steps": total_steps,
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": grad_accum,
            "lr_schedule": {
                "type": "warmup_stable_decay",
                "warmup_steps": warmup,
                "stable_steps": stable,
                "decay_steps": decay,
                "max_lr": round(lr, 6),
                "min_lr": round(lr / 100, 8),
            },
        }
    )

    cfg["validation"] = {"every_steps": max(50, total_steps // 10), "num_samples": 2048}
    cfg["logging"]["run_name"] = name
    cfg["checkpoint"]["save_dir"] = f"models/scaling/{name}"
    return cfg


def parse_metrics(output):
    """Parse final val loss and move accuracy from training output."""
    val_losses, move_accs = [], []
    for line in output.split("\n"):
        m = re.search(r"Val loss:\s*([\d.]+)\s*\|\s*Move acc:\s*([\d.]+)", line)
        if m:
            val_losses.append(float(m.group(1)))
            move_accs.append(float(m.group(2)))

    # Get final training loss
    train_loss = None
    for line in reversed(output.split("\n")):
        m = re.search(r"Loss:\s*([\d.]+)", line)
        if m:
            train_loss = float(m.group(1))
            break

    # Get throughput
    tps = None
    for line in reversed(output.split("\n")):
        m = re.search(r"Tokens/s:\s*([\d.]+)K", line)
        if m:
            tps = float(m.group(1)) * 1000
            break

    # Get param count
    params = None
    for line in output.split("\n"):
        m = re.search(r"Number of parameters:\s*([\d.]+[MBK]?)", line)
        if m:
            params = m.group(1)
            break

    return {
        "val_losses": val_losses,
        "move_accs": move_accs,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "final_move_acc": move_accs[-1] if move_accs else None,
        "final_train_loss": train_loss,
        "throughput_tps": tps,
        "params": params,
    }


def run_experiment(name, config):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    cmd = [
        ".venv/bin/torchrun",
        "--nproc_per_node",
        "8",
        "picotron/train.py",
        "--config",
        config_path,
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["FLASH_ATTEN"] = "0"

    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(
        f"  Config: {json.dumps({k: config['training'][k] for k in ['total_train_steps', 'micro_batch_size']}, indent=None)}"
    )
    print(f"  LR: {config['training']['lr_schedule']['max_lr']}")
    print(f"{'=' * 60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    elapsed = time.time() - start

    output = result.stdout + result.stderr
    os.unlink(config_path)

    if result.returncode != 0:
        # Print last 20 lines of output for debugging
        lines = output.strip().split("\n")
        print(f"  FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        for line in lines[-20:]:
            print(f"  | {line}")
        return {"status": "failed", "elapsed": elapsed, "error": lines[-5:] if lines else []}

    metrics = parse_metrics(output)
    print(
        f"  Done in {elapsed:.0f}s | val_loss={metrics['final_val_loss']} | move_acc={metrics['final_move_acc']} | params={metrics['params']}"
    )
    return {"status": "ok", "elapsed": elapsed, **metrics}


def main():
    results = {}
    for compute_label, model_size, total_tokens in EXPERIMENTS:
        name = f"scale_{compute_label}_{model_size}"
        config = make_config(name, model_size, total_tokens)
        actual_tokens = (
            config["training"]["total_train_steps"]
            * config["training"]["micro_batch_size"]
            * config["training"]["gradient_accumulation_steps"]
            * 8
            * 1024
        )
        result = run_experiment(name, config)
        results[name] = {
            "compute_budget": compute_label,
            "model_size": model_size,
            "target_tokens": total_tokens,
            "actual_tokens": actual_tokens,
            "flops": 6 * int(model_size.rstrip("M")) * 1e6 * actual_tokens,
            "config": {
                "hidden_size": config["model"]["hidden_size"],
                "num_layers": config["model"]["num_hidden_layers"],
                "lr": config["training"]["lr_schedule"]["max_lr"],
                "steps": config["training"]["total_train_steps"],
                "batch_tokens": config["training"]["micro_batch_size"]
                * config["training"]["gradient_accumulation_steps"]
                * 8
                * 1024,
            },
            **result,
        }

        # Save incrementally
        with open("results/scaling_law_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Print summary table
    print("\n\n" + "=" * 80)
    print("SCALING LAW RESULTS")
    print("=" * 80)
    for compute in ["1e17", "3e17", "1e18"]:
        print(f"\n--- Compute: {compute} FLOPs ---")
        print(
            f"{'Name':>25} {'Params':>8} {'Tokens':>8} {'T/N':>6} {'ValLoss':>8} {'MoveAcc':>8} {'Time':>6}"
        )
        for name, r in sorted(results.items()):
            if r["compute_budget"] != compute:
                continue
            tn = r["actual_tokens"] / (int(r["model_size"].rstrip("M")) * 1e6)
            print(
                f"{name:>25} {r['params'] or '?':>8} {r['actual_tokens'] / 1e9:>6.2f}B {tn:>5.0f} {r.get('final_val_loss', '?'):>8} {r.get('final_move_acc', '?'):>8} {r.get('elapsed', 0) / 60:>5.1f}m"
            )

    print(f"\nResults saved to results/scaling_law_results.json")


if __name__ == "__main__":
    main()
