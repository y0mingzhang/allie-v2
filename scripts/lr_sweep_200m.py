#!/usr/bin/env python3
"""Quick LR sweep for 200M model: 2000 steps each."""

import json, subprocess, tempfile, time, re, os, copy

BASE = {
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
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 14,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
    },
    "training": {
        "seed": 42,
        "total_train_steps": 2000,
        "seq_length": 1024,
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 2,
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
    "validation": {"every_steps": 200, "num_samples": 2048},
    "checkpoint": {"save_dir": "", "save_frequency": 999999, "load_path": ""},
    "logging": {"use_wandb": False, "project_name": "chess-v3-lr200m", "run_name": ""},
    "environment": {"OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "FLASH_ATTEN": "0"},
}

results = {}
for lr in [0.012, 0.02, 0.025]:
    cfg = copy.deepcopy(BASE)
    cfg["training"]["lr_schedule"] = {
        "type": "warmup_stable_decay",
        "warmup_steps": 50,
        "stable_steps": 1750,
        "decay_steps": 200,
        "max_lr": lr,
        "min_lr": lr / 100,
    }
    cfg["logging"]["run_name"] = f"lr200m_{lr}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        path = f.name

    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "FLASH_ATTEN": "0"})

    print(f"\nRunning LR={lr}...", flush=True)
    start = time.time()
    r = subprocess.run(
        [".venv/bin/torchrun", "--nproc_per_node", "4", "picotron/train.py", "--config", path],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    elapsed = time.time() - start
    os.unlink(path)

    output = r.stdout + r.stderr
    vals = re.findall(
        r"Val loss:\s*([\d.]+)\s*\|\s*Move acc:\s*([\d.]+)\s*\|\s*Bits/move:\s*([\d.]+)", output
    )
    if vals:
        last = vals[-1]
        print(
            f"  LR={lr}: val_loss={last[0]}, move_acc={last[1]}, bits/move={last[2]} ({elapsed:.0f}s)",
            flush=True,
        )
        results[lr] = {"val_loss": last[0], "move_acc": last[1], "bits_per_move": last[2]}
    else:
        print(f"  LR={lr}: FAILED ({elapsed:.0f}s)", flush=True)
        lines = output.strip().split("\n")
        for l in lines[-5:]:
            print(f"  | {l}", flush=True)

print("\n\nSummary:", flush=True)
for lr, r in sorted(results.items()):
    print(f"  LR={lr}: {r}", flush=True)
