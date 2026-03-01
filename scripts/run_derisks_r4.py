#!/usr/bin/env python3
"""R4: Data quality experiments with engine data mixing and Stockfish eval.

Tests: baseline vs mixed (human+engine) data, then plays Stockfish to measure actual strength.
Uses 40M Qwen3 model, 4062 steps each, 4 GPUs.
"""

import json, math, os, re, subprocess, tempfile, time, copy

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
        "total_train_steps": 8124,
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
            "warmup_steps": 100,
            "stable_steps": 7212,
            "decay_steps": 812,
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
    "validation": {"every_steps": 400, "num_samples": 2048},
    "checkpoint": {"save_dir": "", "save_frequency": 8124, "load_path": ""},
    "logging": {"use_wandb": False, "project_name": "chess-v3-data", "run_name": ""},
    "environment": {"OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "FLASH_ATTEN": "0"},
}

EXPERIMENTS = {
    "baseline": {},
    "mixed_engine": {
        "dataset": {
            "train_glob": "data/tokens_v2/mixed_v1/train.npy",
            "val_glob": "data/tokens_v2/mixed_v1/val.npy",
        }
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
    val_losses, move_accs, bits_per_move = [], [], []
    for line in output.split("\n"):
        m = re.search(r"Val loss:\s*([\d.]+)\s*\|\s*Move acc:\s*([\d.]+)", line)
        if m:
            val_losses.append(float(m.group(1)))
            move_accs.append(float(m.group(2)))
        m2 = re.search(r"Bits/move:\s*([\d.]+)", line)
        if m2:
            bits_per_move.append(float(m2.group(1)))
    train_loss = params = None
    for line in reversed(output.split("\n")):
        m = re.search(r"Loss:\s*([\d.]+)", line)
        if m:
            train_loss = float(m.group(1))
            break
    for line in output.split("\n"):
        m = re.search(r"Number of parameters:\s*([\d.]+[MBK]?)", line)
        if m:
            params = m.group(1)
            break
    return {
        "val_losses": val_losses,
        "move_accs": move_accs,
        "bits_per_move": bits_per_move,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "final_move_acc": move_accs[-1] if move_accs else None,
        "final_bits_per_move": bits_per_move[-1] if bits_per_move else None,
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
        "4",
        "picotron/train.py",
        "--config",
        config_path,
    ]
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "FLASH_ATTEN": "0"})
    print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
    elapsed = time.time() - start
    output = result.stdout + result.stderr
    os.unlink(config_path)
    if result.returncode != 0:
        lines = output.strip().split("\n")
        print(f"  FAILED after {elapsed:.0f}s", flush=True)
        for line in lines[-15:]:
            print(f"  | {line}", flush=True)
        return {"status": "failed", "elapsed": elapsed}
    metrics = parse_metrics(output)
    bpm = f" | bits/move={metrics['final_bits_per_move']}" if metrics["final_bits_per_move"] else ""
    print(
        f"  Done in {elapsed:.0f}s | val_loss={metrics['final_val_loss']} | move_acc={metrics['final_move_acc']}{bpm}",
        flush=True,
    )
    return {"status": "ok", "elapsed": elapsed, **metrics}


def main():
    results = {}
    for exp_name, overrides in EXPERIMENTS.items():
        name = f"r4_{exp_name}"
        config = deep_update(BASE_CONFIG, overrides)
        config["logging"]["run_name"] = name
        config["checkpoint"]["save_dir"] = f"models/derisk_r4/{name}"
        result = run_experiment(name, config)
        results[name] = {"experiment": exp_name, **result}
        with open("results/derisk_results_r4.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}\nR4 DATA QUALITY RESULTS\n{'=' * 70}", flush=True)
    print(f"{'Name':>25} {'ValLoss':>8} {'MoveAcc':>8} {'Bits/M':>8} {'Time':>6}")
    for name, r in sorted(results.items(), key=lambda x: x[1].get("final_val_loss", 99)):
        bpm = r.get("final_bits_per_move", "?")
        print(
            f"{name:>25} {r.get('final_val_loss', '?'):>8} {r.get('final_move_acc', '?'):>8} {bpm:>8} {r.get('elapsed', 0) / 60:>5.1f}m"
        )


if __name__ == "__main__":
    main()
