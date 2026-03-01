#!/usr/bin/env python3
"""Architecture ablation experiments at fixed compute (1e18 FLOPs, ~40M params).

Tests nanogpt-speedrun ideas and other architectural changes:
1. baseline: Qwen3 40M (our scaling law winner)
2. relu2: replace SiLU with ReLU²
3. embed_shortcut: add embedding residual to every layer (x0 shortcut)
4. zero_init_proj: zero-init output projections (muP-style)
5. deeper_narrow: same params but 16L x h=416 instead of 12L x h=512
6. gpt2_style: LayerNorm + learned position embeddings (no RoPE, no GQA)
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
        "hidden_size": 512,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
    },
    "training": {
        "seed": 42,
        "total_train_steps": 4062,
        "seq_length": 1024,
        "micro_batch_size": 64,
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
        "lr_schedule": {
            "type": "warmup_stable_decay",
            "warmup_steps": 50,
            "stable_steps": 3606,
            "decay_steps": 406,
            "max_lr": 0.03,
            "min_lr": 0.0003,
        },
    },
    "dataset": {
        "name": "tokens",
        "train_glob": "data/tokens_v2/full_v1/train.npy",
        "val_glob": "data/tokens_v2/full_v1/val.npy",
        "num_workers": 4,
    },
    "validation": {"every_steps": 200, "num_samples": 2048},
    "checkpoint": {"save_dir": "", "save_frequency": 999999, "load_path": ""},
    "logging": {"use_wandb": False, "project_name": "chess-v3-arch", "run_name": ""},
    "environment": {"OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "FLASH_ATTEN": "0"},
}


EXPERIMENTS = {
    "baseline": {},  # no changes
    "relu2": {"model": {"hidden_act": "relu2"}},
    "embed_shortcut": {"model": {"embed_shortcut": True}},
    "zero_init_proj": {"model": {"zero_init_proj": True}},
    "deeper_narrow": {
        "model": {
            "hidden_size": 416,
            "intermediate_size": 1248,
            "num_hidden_layers": 16,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 52,
        },
    },
    "wider_shallow": {
        "model": {
            "hidden_size": 640,
            "intermediate_size": 1920,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 80,
        },
    },
}


def deep_update(base, override):
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def parse_metrics(output):
    val_losses, move_accs = [], []
    for line in output.split("\n"):
        m = re.search(r"Val loss:\s*([\d.]+)\s*\|\s*Move acc:\s*([\d.]+)", line)
        if m:
            val_losses.append(float(m.group(1)))
            move_accs.append(float(m.group(2)))
    train_loss = None
    for line in reversed(output.split("\n")):
        m = re.search(r"Loss:\s*([\d.]+)", line)
        if m:
            train_loss = float(m.group(1))
            break
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
    env.update({"OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "FLASH_ATTEN": "0"})

    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(
        f"  Steps: {config['training']['total_train_steps']}, LR: {config['training']['lr_schedule']['max_lr']}"
    )
    model_cfg = {
        k: config["model"].get(k)
        for k in ["hidden_size", "num_hidden_layers", "head_dim"]
        if k in config["model"]
    }
    print(f"  Model: {model_cfg}")
    print(f"{'=' * 60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    elapsed = time.time() - start
    output = result.stdout + result.stderr
    os.unlink(config_path)

    if result.returncode != 0:
        lines = output.strip().split("\n")
        print(f"  FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        for line in lines[-15:]:
            print(f"  | {line}")
        return {"status": "failed", "elapsed": elapsed}

    metrics = parse_metrics(output)
    print(
        f"  Done in {elapsed:.0f}s | val_loss={metrics['final_val_loss']} | move_acc={metrics['final_move_acc']} | params={metrics['params']}"
    )
    return {"status": "ok", "elapsed": elapsed, **metrics}


def main():
    results = {}
    for exp_name, overrides in EXPERIMENTS.items():
        name = f"arch_{exp_name}"
        config = deep_update(BASE_CONFIG, overrides)
        config["logging"]["run_name"] = name
        config["checkpoint"]["save_dir"] = f"models/arch/{name}"

        result = run_experiment(name, config)
        results[name] = {"experiment": exp_name, "overrides": overrides, **result}

        with open("results/arch_ablation_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n\n" + "=" * 70)
    print("ARCHITECTURE ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Name':>20} {'Params':>8} {'ValLoss':>8} {'MoveAcc':>8} {'Time':>6}")
    for name, r in sorted(results.items(), key=lambda x: x[1].get("final_val_loss", 99)):
        print(
            f"{name:>20} {r.get('params', '?'):>8} {r.get('final_val_loss', '?'):>8} {r.get('final_move_acc', '?'):>8} {r.get('elapsed', 0) / 60:>5.1f}m"
        )


if __name__ == "__main__":
    main()
