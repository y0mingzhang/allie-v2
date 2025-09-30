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
from collections import OrderedDict
import json
import os
from pathlib import Path
import shutil
import sys
import types

from safetensors.torch import save_file
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from transformers import AutoConfig

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("FLASH_ATTEN", "0")
os.environ.setdefault("CONTEXT_PARALLEL", "0")


def _ensure_flash_attn_stub() -> None:
    try:
        return
    except Exception:
        pass

    for module_name in (
        "flash_attn",
        "flash_attn.flash_attn_interface",
        "flash_attn.layers",
        "flash_attn.layers.rotary",
        "flash_attn.ops",
        "flash_attn.ops.triton",
        "flash_attn.ops.triton.layer_norm",
    ):
        sys.modules.pop(module_name, None)

    flash_attn_root = types.ModuleType("flash_attn")
    interface_module = types.ModuleType("flash_attn.flash_attn_interface")
    rotary_module = types.ModuleType("flash_attn.layers.rotary")
    layer_norm_module = types.ModuleType("flash_attn.ops.triton.layer_norm")

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_emb(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
    ) -> torch.Tensor:
        if interleaved:
            raise NotImplementedError("Interleaved rotary embedding not supported in CPU stub")
        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return (x * cos) + (_rotate_half(x) * sin)

    def _flash_attn_func(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, **_: object
    ) -> torch.Tensor:
        bsz, seqlen, heads, head_dim = q.shape
        q_flat = q.permute(0, 2, 1, 3).reshape(bsz * heads, seqlen, head_dim)
        k_flat = k.permute(0, 2, 1, 3).reshape(bsz * heads, seqlen, head_dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(bsz * heads, seqlen, head_dim)
        out = F.scaled_dot_product_attention(
            q_flat, k_flat, v_flat, attn_mask=None, dropout_p=0.0, is_causal=causal
        )
        out = out.reshape(bsz, heads, seqlen, head_dim).permute(0, 2, 1, 3)
        return out

    def _layer_norm_fn(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        residual: torch.Tensor | None = None,
        eps: float = 1e-5,
        dropout_p: float = 0.0,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
        return_dropout_mask: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        if residual is not None:
            hidden_states = hidden_states + residual
        working = hidden_states.float()
        if is_rms_norm:
            variance = working.pow(2).mean(dim=-1, keepdim=True)
            working = working * torch.rsqrt(variance + eps)
        else:
            mean = working.mean(dim=-1, keepdim=True)
            variance = working.var(dim=-1, unbiased=False, keepdim=True)
            working = (working - mean) * torch.rsqrt(variance + eps)
        out = (weight * working.to(weight.dtype)).to(hidden_states.dtype)
        if bias is not None:
            out = out + bias.to(out.dtype)
        if return_dropout_mask:
            return out, None
        return out

    interface_module.flash_attn_func = _flash_attn_func
    rotary_module.apply_rotary_emb = _apply_rotary_emb
    layer_norm_module.layer_norm_fn = _layer_norm_fn

    flash_attn_root.flash_attn_interface = interface_module
    flash_attn_root.layers = types.SimpleNamespace(rotary=rotary_module)
    flash_attn_root.ops = types.SimpleNamespace(
        triton=types.SimpleNamespace(layer_norm=layer_norm_module)
    )

    sys.modules["flash_attn"] = flash_attn_root
    sys.modules["flash_attn.flash_attn_interface"] = interface_module
    sys.modules["flash_attn.layers"] = types.SimpleNamespace(rotary=rotary_module)
    sys.modules["flash_attn.layers.rotary"] = rotary_module
    sys.modules["flash_attn.ops"] = types.SimpleNamespace(
        triton=types.SimpleNamespace(layer_norm=layer_norm_module)
    )
    sys.modules["flash_attn.ops.triton"] = types.SimpleNamespace(layer_norm=layer_norm_module)
    sys.modules["flash_attn.ops.triton.layer_norm"] = layer_norm_module


_ensure_flash_attn_stub()

from picotron.model import Llama, Qwen3Model  # noqa: E402
from picotron.process_group_manager import setup_process_group_manager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to Picotron JSON config used for training"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the Picotron checkpoint (.pth) to export"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write Hugging Face files")
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Optional directory containing tokenizer files to copy",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Torch dtype to save in config",
    )
    return parser.parse_args()


def load_training_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_base_model_name(model_section: dict) -> str:
    return model_section.get("hf_base", model_section.get("name"))


def _apply_model_overrides(model_cfg: AutoConfig, overrides: dict, train_cfg: dict) -> None:
    simple_overrides = (
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_size",
        "intermediate_size",
        "rope_theta",
        "rms_norm_eps",
        "layer_types",
        "head_dim",
        "attention_bias",
        "attention_dropout",
        "use_sliding_window",
        "sliding_window",
        "max_window_layers",
        "initializer_range",
    )
    for key in simple_overrides:
        if key in overrides and overrides[key] is not None:
            setattr(model_cfg, key, overrides[key])

    if overrides.get("vocab_size") is not None:
        model_cfg.vocab_size = overrides["vocab_size"]

    if overrides.get("tie_word_embeddings") is not None:
        model_cfg.tie_word_embeddings = overrides["tie_word_embeddings"]
    elif getattr(model_cfg, "model_type", "") in {"qwen", "qwen2", "qwen3"}:
        # Picotron checkpoints do not tie embeddings by construction.
        model_cfg.tie_word_embeddings = False

    max_pos = overrides.get("max_position_embeddings")
    if max_pos is None:
        max_pos = train_cfg.get("seq_length", getattr(model_cfg, "max_position_embeddings", None))
    if max_pos is not None:
        model_cfg.max_position_embeddings = max_pos


def build_model_config(config_dict: dict) -> AutoConfig:
    model_section = config_dict["model"]
    train_section = config_dict.get("training", {})

    model_cfg = AutoConfig.from_pretrained(_get_base_model_name(model_section))

    _apply_model_overrides(model_cfg, model_section, train_section)

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
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def instantiate_model(model_cfg: AutoConfig):
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("FLASH_ATTEN", "0")
    model_type = getattr(model_cfg, "model_type", "")
    if model_type == "qwen3":
        model_cls = Qwen3Model
    elif model_type in {"llama", "llama2", "llama3"}:
        model_cls = Llama
    else:
        raise ValueError(f"Unsupported model type for export: {model_type!r}")
    model = model_cls(config=model_cfg)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'model' key. Did you pass the Picotron .pth file?")
    if any(key.startswith("_orig_mod.") for key in state_dict):
        prefix = "_orig_mod."
        state_dict = OrderedDict(
            (key[len(prefix) :], value) if key.startswith(prefix) else (key, value)
            for key, value in state_dict.items()
        )
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
    key = key.replace("q_norm.", "q_norm.")
    key = key.replace("k_norm.", "k_norm.")
    return key


def convert_state_dict_to_hf(model: torch.nn.Module) -> dict:
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
    model_cfg.dtype = dtype
    model_cfg.save_pretrained(out_path)

    if tokenizer_dir is not None:
        tok_src = Path(tokenizer_dir)
        if not tok_src.exists() or not tok_src.is_dir():
            raise ValueError(
                f"Tokenizer directory '{tokenizer_dir}' does not exist or is not a directory"
            )
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
