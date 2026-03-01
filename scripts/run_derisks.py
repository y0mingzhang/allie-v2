#!/usr/bin/env python3
"""Run derisk experiments sequentially. Each experiment is a ~3-5 min training run."""
import json
import subprocess
import sys
import os
import copy
import re
import time

BASE_CONFIG = "configs/derisk/base_tiny.json"

def load_base():
    with open(BASE_CONFIG) as f:
        return json.load(f)

EXPERIMENTS = {
    "base_tiny": {},
    "tie_embed": {
        "model": {"tie_word_embeddings": True},
    },
    "move_weight_3x": {
        "training": {"move_loss_weight": 3.0},
    },
    "move_weight_5x": {
        "training": {"move_loss_weight": 5.0},
    },
    "higher_lr_2x": {
        "training": {"lr_schedule": {"max_lr": 2e-2}},
    },
    "lower_lr_half": {
        "training": {"lr_schedule": {"max_lr": 5e-3}},
    },
    "z_loss": {
        "training": {"z_loss_coeff": 1e-4},
    },
    "deeper_8L": {
        "model": {"num_hidden_layers": 8},
    },
}

def deep_update(base, overlay):
    result = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def make_config(name, overrides):
    cfg = load_base()
    cfg = deep_update(cfg, overrides)
    cfg["logging"]["run_name"] = name
    cfg["checkpoint"]["save_dir"] = f"models/derisk/{name}"
    out = f"configs/derisk/{name}.json"
    with open(out, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    return out


def parse_metrics(output):
    """Extract final training loss, val loss, and move accuracy from output."""
    lines = output.split("\n")
    last_loss = None
    last_val_loss = None
    last_move_acc = None
    losses_at_steps = {}

    for line in lines:
        m = re.search(r"Step:\s*(\d+)\s*\| Loss:\s*([\d.]+)", line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            last_loss = loss
            losses_at_steps[step] = loss

        m = re.search(r"Val loss:\s*([\d.]+)\s*\| Move acc:\s*([\d.]+)", line)
        if m:
            last_val_loss = float(m.group(1))
            last_move_acc = float(m.group(2))

    # Get loss at specific checkpoints for comparison
    loss_100 = losses_at_steps.get(100)
    loss_500 = losses_at_steps.get(500)
    loss_1000 = losses_at_steps.get(1000)

    return {
        "final_loss": last_loss,
        "val_loss": last_val_loss,
        "move_acc": last_move_acc,
        "loss@100": loss_100,
        "loss@500": loss_500,
        "loss@1000": loss_1000,
    }


def run_experiment(name, config_path):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    env = os.environ.copy()
    env["FLASH_ATTEN"] = "0"
    env["WANDB_MODE"] = "offline"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["HF_HOME"] = os.path.expanduser("~/user_data/chess-v3/.cache/huggingface")
    hf_token_path = os.path.expanduser("~/secrets/hf-token")
    env["HF_TOKEN"] = open(hf_token_path).read().strip() if os.path.exists(hf_token_path) else "dummy"
    torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
    result = subprocess.run(
        [torchrun, "--nproc_per_node", "8", "picotron/train.py", "--config", config_path],
        env=env, timeout=600, capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr
    ok = result.returncode == 0
    metrics = parse_metrics(output) if ok else {}
    status = "OK" if ok else f"FAILED (rc={result.returncode})"
    print(f"  {status} in {elapsed:.0f}s")
    if metrics:
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v}")
    if not ok:
        # Print last 20 lines of error
        for line in output.strip().split("\n")[-20:]:
            print(f"  ERR: {line}")
    return ok, metrics, elapsed


def main():
    experiments = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS.keys())
    results = {}
    for name in experiments:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            continue
        config_path = make_config(name, EXPERIMENTS[name])
        ok, metrics, elapsed = run_experiment(name, config_path)
        results[name] = {"ok": ok, "metrics": metrics, "elapsed": elapsed}

    print(f"\n{'='*70}")
    print("DERISK RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'experiment':<18} {'loss@100':>10} {'loss@500':>10} {'loss@1000':>10} {'val_loss':>10} {'move_acc':>10} {'time':>6}")
    print("-" * 70)
    for name, r in results.items():
        m = r["metrics"]
        if not r["ok"]:
            print(f"{name:<18} {'FAILED':>10}")
            continue
        print(
            f"{name:<18} "
            f"{m.get('loss@100', 'n/a'):>10} "
            f"{m.get('loss@500', 'n/a'):>10} "
            f"{m.get('loss@1000', 'n/a'):>10} "
            f"{m.get('val_loss', 'n/a'):>10} "
            f"{m.get('move_acc', 'n/a'):>10} "
            f"{r['elapsed']:>5.0f}s"
        )

    # Save results to JSON
    out_path = "results/derisk_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
