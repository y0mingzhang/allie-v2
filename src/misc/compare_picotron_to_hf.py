#!/usr/bin/env python
"""Compare Picotron and Hugging Face checkpoints for consistency.

The script loads a Picotron checkpoint (.pth) alongside the Hugging Face export
directory produced by ``picotron/picotron/export_to_hf.py`` and verifies that
the tensors match once the appropriate key remapping has been applied.

For every parameter we report the maximum absolute difference and the relative
error. If differences exceed the configured tolerances the script exits with a
non-zero status code.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

from safetensors.torch import load_file
import torch


@dataclass
class Difference:
    name: str
    max_abs: float
    max_rel: float


def picotron_to_hf_key(key: str) -> str:
    key = key.replace("decoder_layers.", "model.layers.")
    key = key.replace("embedding.", "model.embed_tokens.")
    key = key.replace("final_norm.", "model.norm.")
    key = key.replace("final_proj.", "lm_head.")
    key = key.replace("attention.", "self_attn.")
    key = key.replace("out_proj", "o_proj")
    key = key.replace("q_norm.", "q_norm.")
    key = key.replace("k_norm.", "k_norm.")
    return key


def strip_orig_mod_prefix(state: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state.keys()):
        return state
    prefix = "_orig_mod."
    return OrderedDict(
        (key[len(prefix) :], value) if key.startswith(prefix) else (key, value)
        for key, value in state.items()
    )


def load_picotron_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model")
    if state is None:
        raise ValueError(f"Checkpoint '{path}' missing 'model' key")
    if not isinstance(state, OrderedDict):
        state = OrderedDict(state)
    state = strip_orig_mod_prefix(state)
    return {picotron_to_hf_key(k): v.detach().cpu() for k, v in state.items()}


def load_hf_state(model_dir: Path) -> dict[str, torch.Tensor]:
    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} does not exist")
    state = load_file(str(weights_path))
    return {k: v.detach().cpu() for k, v in state.items()}


def compare_states(
    picotron_state: dict[str, torch.Tensor],
    hf_state: dict[str, torch.Tensor],
    atol: float,
    rtol: float,
) -> list[Difference]:
    differences: list[Difference] = []
    for key, picotron_tensor in picotron_state.items():
        if key not in hf_state:
            raise KeyError(f"HF state missing key '{key}'")
        hf_tensor = hf_state[key]
        if picotron_tensor.shape != hf_tensor.shape:
            raise ValueError(
                f"Shape mismatch for '{key}': {picotron_tensor.shape} vs {hf_tensor.shape}"
            )
        diff = (picotron_tensor - hf_tensor).float()
        abs_diff = diff.abs()
        max_abs = abs_diff.max().item()
        denom = hf_tensor.abs().clamp_min(1e-8)
        rel_diff = abs_diff / denom
        max_rel = rel_diff.max().item()
        differences.append(Difference(key, max_abs, max_rel))
        if max_abs > atol and max_rel > rtol:
            print(f"[ERROR] {key}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        else:
            print(f"[ OK ] {key}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--picotron-ckpt", required=True, help="Path to the Picotron .pth checkpoint"
    )
    parser.add_argument(
        "--hf-dir",
        required=True,
        help="Directory containing the Hugging Face export (model.safetensors/config.json)",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--summary", action="store_true", help="Only print a final JSON summary")

    args = parser.parse_args()

    picotron_state = load_picotron_state(Path(args.picotron_ckpt))
    hf_state = load_hf_state(Path(args.hf_dir))

    differences = compare_states(picotron_state, hf_state, args.atol, args.rtol)

    failed = [d for d in differences if d.max_abs > args.atol and d.max_rel > args.rtol]
    if args.summary:
        summary = {
            "num_parameters": len(differences),
            "max_abs_diff": max((d.max_abs for d in differences), default=0.0),
            "max_rel_diff": max((d.max_rel for d in differences), default=0.0),
            "num_failures": len(failed),
        }
        print(json.dumps(summary, indent=2))
    else:
        print(f"Compared {len(differences)} tensors")
        if failed:
            print(f"{len(failed)} tensors exceeded tolerances (atol={args.atol}, rtol={args.rtol})")
        else:
            print("All tensors matched within tolerances")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
