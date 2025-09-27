#!/usr/bin/env python
"""Convert a Picotron checkpoint to Hugging Face format.

This script assumes tensor / pipeline / context parallel sizes of 1 when loading the
checkpoint. Data parallel checkpoints saved from rank 0 are acceptable because the
state dict is identical across DP ranks.

Example usage:

    python scripts/export_to_hf.py \
        --config configs/llama-3-1B-57B.json \
        --checkpoint models/Llama-3-/weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth \
        --output-dir export/llama-3-1B-hf \
        --tokenizer-dir data/tokenizer

The output directory will contain `model.safetensors` and `config.json`. If a
tokenizer directory is provided, its files are copied alongside the model files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import AutoConfig

from picotron.model import Llama
from picotron.process_group_manager import setup_process_group_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to Picotron JSON config used for training")
    parser.add_argument("--checkpoint", required=True, help="Path to the Picotron checkpoint (.pth) to export")
    parser.add_argument("--output-dir", required=True, help="Directory to write Hugging Face files")
    parser.add_argument("--tokenizer-dir", default=None, help="Optional directory containing tokenizer files to copy")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="Torch dtype to save in config")
    return parser.parse_args()


def load_training_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_model_config(config_dict: dict) -> AutoConfig:
    model_cfg = AutoConfig.from_pretrained(config_dict["model"]["name"])

    overrides = config_dict["model"]
    if "num_hidden_layers" in overrides:
        model_cfg.num_hidden_layers = overrides["num_hidden_layers"]
    if "num_attention_heads" in overrides:
        model_cfg.num_attention_heads = overrides["num_attention_heads"]
    if "num_key_value_heads" in overrides:
        model_cfg.num_key_value_heads = overrides["num_key_value_heads"]
    if overrides.get("vocab_size") is not None:
        model_cfg.vocab_size = overrides["vocab_size"]

    train_cfg = config_dict["training"]
    model_cfg.max_position_embeddings = train_cfg["seq_length"]

    return model_cfg


def ensure_single_parallelism(config_dict: dict) -> None:
    dist_cfg = config_dict["distributed"]
    for key in ("tp_size", "pp_size", "cp_size"):
        if dist_cfg.get(key, 1) != 1:
            raise ValueError(f"Export currently supports {key}=1 only (found {dist_cfg.get(key)})")


def init_distributed() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def instantiate_model(model_cfg: AutoConfig) -> Llama:
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("FLASH_ATTEN", "0")
    model = Llama(config=model_cfg)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'model' key. Did you pass the Picotron .pth file?")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise ValueError(f"Model state dict missing keys: {missing}")
    if unexpected:
        raise ValueError(f"Model state dict had unexpected keys: {unexpected}")


def picotron_to_hf_key(key: str) -> str:
    key = key.replace("decoder_layers.", "model.layers.")
    key = key.replace("embedding.", "model.embed_tokens.")
    key = key.replace("final_norm.", "model.norm.")
    key = key.replace("final_proj.", "lm_head.")
    key = key.replace("attention.", "self_attn.")
    key = key.replace("out_proj", "o_proj")
    return key


def convert_state_dict_to_hf(model: Llama) -> dict:
    picotron_state = model.state_dict()
    hf_state = {}
    for key, tensor in picotron_state.items():
        hf_key = picotron_to_hf_key(key)
        hf_state[hf_key] = tensor.detach().cpu()
    return hf_state


def save_hf_artifacts(
    model_cfg: AutoConfig,
    hf_state: dict,
    output_dir: str,
    tokenizer_dir: str | None,
    dtype: str,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    save_file(hf_state, out_path / "model.safetensors")

    # Persist config.json using transformers serializer
    model_cfg.torch_dtype = dtype
    model_cfg.save_pretrained(out_path)

    if tokenizer_dir is not None:
        tok_src = Path(tokenizer_dir)
        if not tok_src.exists() or not tok_src.is_dir():
            raise ValueError(f"Tokenizer directory '{tokenizer_dir}' does not exist or is not a directory")
        for item in tok_src.iterdir():
            dest = out_path / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)


def main() -> None:
    args = parse_args()

    config_dict = load_training_config(args.config)
    ensure_single_parallelism(config_dict)

    init_distributed()
    setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=1)

    try:
        model_cfg = build_model_config(config_dict)
        model = instantiate_model(model_cfg)
        load_checkpoint(model, args.checkpoint)
        hf_state = convert_state_dict_to_hf(model)
        save_hf_artifacts(model_cfg, hf_state, args.output_dir, args.tokenizer_dir, args.dtype)
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()

