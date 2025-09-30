#!/usr/bin/env python
"""Convert a Picotron checkpoint to Hugging Face format.

This script assumes tensor / pipeline / context parallel sizes of 1 when loading the
checkpoint. Data parallel checkpoints saved from rank 0 are acceptable because the
state dict is identical across DP ranks.

Example usage:

    python src/tools/hf/export_checkpoint.py \
        --config configs/main_runs_v2/qwen-3-4b-58b.json \
        --checkpoint models/main_runs/qwen-3-4b-58b/weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth \
        --output-dir export/qwen-3-4b-58b \
        --tokenizer-dir data/tokenizer \
        --dtype bfloat16

The output directory will contain `model.safetensors` and `config.json`. If a
tokenizer directory is provided, its files are copied alongside the model files.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import logging
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

try:
    from huggingface_hub import HfApi, create_repo
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Configure for CPU-only export
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("FLASH_ATTEN", "0")
os.environ.setdefault("CONTEXT_PARALLEL", "0")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants for key mappings
PICOTRON_TO_HF_KEY_MAP = {
    "decoder_layers.": "model.layers.",
    "embedding.": "model.embed_tokens.",
    "final_norm.": "model.norm.",
    "final_proj.": "lm_head.",
    "attention.": "self_attn.",
    "out_proj": "o_proj",
}

TORCH_COMPILE_PREFIX = "_orig_mod."


def _ensure_flash_attn_stub() -> None:
    """Create a CPU-compatible stub for flash_attn to avoid CUDA initialization.
    
    This stub implements basic versions of flash_attn functions that work on CPU,
    allowing the model to be loaded without GPU access.
    """
    # Remove any existing flash_attn modules
    module_names = [
        "flash_attn",
        "flash_attn.flash_attn_interface",
        "flash_attn.layers",
        "flash_attn.layers.rotary",
        "flash_attn.ops",
        "flash_attn.ops.triton",
        "flash_attn.ops.triton.layer_norm",
    ]
    for module_name in module_names:
        sys.modules.pop(module_name, None)

    # Create stub modules
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

    # Attach functions to modules
    interface_module.flash_attn_func = _flash_attn_func
    rotary_module.apply_rotary_emb = _apply_rotary_emb
    layer_norm_module.layer_norm_fn = _layer_norm_fn

    # Build module hierarchy
    layers_namespace = types.SimpleNamespace(rotary=rotary_module)
    triton_namespace = types.SimpleNamespace(layer_norm=layer_norm_module)
    ops_namespace = types.SimpleNamespace(triton=triton_namespace)
    
    flash_attn_root.flash_attn_interface = interface_module
    flash_attn_root.layers = layers_namespace
    flash_attn_root.ops = ops_namespace

    # Register all modules in sys.modules
    sys.modules["flash_attn"] = flash_attn_root
    sys.modules["flash_attn.flash_attn_interface"] = interface_module
    sys.modules["flash_attn.layers"] = layers_namespace
    sys.modules["flash_attn.layers.rotary"] = rotary_module
    sys.modules["flash_attn.ops"] = ops_namespace
    sys.modules["flash_attn.ops.triton"] = triton_namespace
    sys.modules["flash_attn.ops.triton.layer_norm"] = layer_norm_module


_ensure_flash_attn_stub()

from picotron.model import Llama, Qwen3Model  # noqa: E402
from picotron.process_group_manager import setup_process_group_manager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        help="Torch dtype to save in config (default: bfloat16)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the exported model to HuggingFace Hub after export",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HuggingFace repository ID (e.g., 'username/model-name'). Required if --upload is set.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace repository private (default: public)",
    )
    return parser.parse_args()


def load_training_config(path: str) -> dict:
    """Load training configuration from JSON file."""
    logger.info(f"Loading training config from {path}")
    with open(path) as f:
        return json.load(f)


def _get_base_model_name(model_section: dict) -> str:
    """Get the HuggingFace base model name from config."""
    return model_section.get("hf_base", model_section.get("name"))


def _apply_model_overrides(model_cfg: AutoConfig, overrides: dict, train_cfg: dict) -> None:
    """Apply model architecture overrides from Picotron config to HF config."""
    # Simple overrides that can be directly copied
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

    # Vocab size override
    if overrides.get("vocab_size") is not None:
        model_cfg.vocab_size = overrides["vocab_size"]

    # Embedding tying (Picotron doesn't tie embeddings for Qwen models)
    if overrides.get("tie_word_embeddings") is not None:
        model_cfg.tie_word_embeddings = overrides["tie_word_embeddings"]
    elif getattr(model_cfg, "model_type", "") in {"qwen", "qwen2", "qwen3"}:
        model_cfg.tie_word_embeddings = False

    # Max position embeddings (use seq_length from training config if not specified)
    max_pos = overrides.get("max_position_embeddings")
    if max_pos is None:
        max_pos = train_cfg.get("seq_length", getattr(model_cfg, "max_position_embeddings", None))
    if max_pos is not None:
        model_cfg.max_position_embeddings = max_pos


def build_model_config(config_dict: dict) -> AutoConfig:
    """Build HuggingFace model config from Picotron training config."""
    model_section = config_dict["model"]
    train_section = config_dict.get("training", {})

    logger.info(f"Loading base model config: {_get_base_model_name(model_section)}")
    model_cfg = AutoConfig.from_pretrained(_get_base_model_name(model_section))

    _apply_model_overrides(model_cfg, model_section, train_section)

    return model_cfg


def ensure_single_parallelism(config_dict: dict) -> None:
    """Ensure the checkpoint was saved with no model parallelism."""
    dist_cfg = config_dict.get("distributed", {})
    for key in ("tp_size", "pp_size", "cp_size"):
        if dist_cfg.get(key, 1) != 1:
            raise ValueError(
                f"Export currently supports {key}=1 only (found {dist_cfg.get(key)}). "
                f"Please use a checkpoint saved without model parallelism."
            )


def init_distributed() -> None:
    """Initialize distributed environment for single process."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)


def shutdown_distributed() -> None:
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def instantiate_model(model_cfg: AutoConfig) -> torch.nn.Module:
    """Instantiate the model based on its type."""
    model_type = getattr(model_cfg, "model_type", "")
    logger.info(f"Instantiating {model_type} model")
    
    if model_type == "qwen3":
        model_cls = Qwen3Model
    elif model_type in {"llama", "llama2", "llama3"}:
        model_cls = Llama
    else:
        raise ValueError(
            f"Unsupported model type for export: {model_type!r}. "
            f"Supported types: qwen3, llama, llama2, llama3"
        )
    
    model = model_cls(config=model_cfg)
    model.eval()
    return model


def _strip_torch_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove torch.compile prefix from state dict keys."""
    if not any(key.startswith(TORCH_COMPILE_PREFIX) for key in state_dict):
        return state_dict
    
    return OrderedDict(
        (key.removeprefix(TORCH_COMPILE_PREFIX), value)
        for key, value in state_dict.items()
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load Picotron checkpoint into model."""
    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    state_dict = checkpoint.get("model")
    if state_dict is None:
        raise ValueError(
            "Checkpoint missing 'model' key. Did you pass the correct Picotron .pth file?"
        )
    
    # Strip torch.compile prefix if present
    state_dict = _strip_torch_compile_prefix(state_dict)
    
    # Sanity check: verify embedding shapes match expected vocab size
    embedding_key = "embedding.weight"
    if embedding_key in state_dict:
        actual_vocab_size = state_dict[embedding_key].shape[0]
        expected_vocab_size = model.embedding.weight.shape[0]
        logger.info(f"Checkpoint embedding shape: {state_dict[embedding_key].shape}")
        logger.info(f"Model expects vocab_size: {expected_vocab_size}")
        
        if actual_vocab_size != expected_vocab_size:
            raise ValueError(
                f"Vocab size mismatch! Checkpoint has {actual_vocab_size} embeddings, "
                f"but model expects {expected_vocab_size}. "
                f"Check that the training config vocab_size matches the checkpoint."
            )
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise ValueError(f"Model state dict missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint (ignoring): {unexpected}")


def picotron_to_hf_key(key: str) -> str:
    """Convert Picotron parameter name to HuggingFace format."""
    for picotron_name, hf_name in PICOTRON_TO_HF_KEY_MAP.items():
        key = key.replace(picotron_name, hf_name)
    return key


def convert_state_dict_to_hf(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Convert model state dict from Picotron to HuggingFace format."""
    logger.info("Converting state dict to HuggingFace format")
    picotron_state = model.state_dict()
    hf_state = {}
    for key, tensor in picotron_state.items():
        hf_key = picotron_to_hf_key(key)
        hf_state[hf_key] = tensor.detach().cpu()
    
    # Sanity check: verify vocab size in converted state
    if "model.embed_tokens.weight" in hf_state:
        vocab_size = hf_state["model.embed_tokens.weight"].shape[0]
        logger.info(f"Converted embedding vocab_size: {vocab_size}")
        if "lm_head.weight" in hf_state:
            lm_head_vocab = hf_state["lm_head.weight"].shape[0]
            if lm_head_vocab != vocab_size:
                logger.warning(
                    f"LM head vocab size ({lm_head_vocab}) differs from embedding vocab size ({vocab_size})"
                )
    
    logger.info(f"Converted {len(hf_state)} parameters")
    return hf_state


def save_hf_artifacts(
    model_cfg: AutoConfig,
    hf_state: dict[str, torch.Tensor],
    output_dir: str,
    tokenizer_dir: str | None,
    dtype: str,
) -> None:
    """Save model, config, and optionally tokenizer to output directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Final sanity check: verify config vocab_size matches actual weights
    if "model.embed_tokens.weight" in hf_state:
        actual_vocab_size = hf_state["model.embed_tokens.weight"].shape[0]
        config_vocab_size = model_cfg.vocab_size
        
        logger.info(f"{'='*60}")
        logger.info(f"SANITY CHECK:")
        logger.info(f"  Config vocab_size: {config_vocab_size}")
        logger.info(f"  Actual embedding weights: {actual_vocab_size}")
        logger.info(f"{'='*60}")
        
        if actual_vocab_size != config_vocab_size:
            raise ValueError(
                f"CRITICAL: Config vocab_size ({config_vocab_size}) does not match "
                f"actual embedding weights ({actual_vocab_size})! "
                f"The exported model would be broken. Export aborted."
            )

    # Save model weights
    logger.info(f"Saving model weights to {out_path / 'model.safetensors'}")
    save_file(hf_state, out_path / "model.safetensors")

    # Save config with correct token IDs
    logger.info(f"Saving config to {out_path / 'config.json'}")
    model_cfg.dtype = dtype
    
    # Set token IDs based on custom chess tokenizer
    # Chess tokenizer only has <bos> and <unk>, no separate EOS/PAD
    # Set all special tokens to BOS token ID
    if model_cfg.vocab_size == 2350:
        model_cfg.bos_token_id = 2348  # <bos>
        model_cfg.eos_token_id = 2348  # No separate EOS, use BOS
        model_cfg.pad_token_id = 2348  # No separate PAD, use BOS
        logger.info(f"Set custom chess tokenizer IDs: bos=eos=pad={model_cfg.bos_token_id}")
    
    model_cfg.save_pretrained(out_path)

    # Copy tokenizer if provided
    if tokenizer_dir is not None:
        logger.info(f"Copying tokenizer from {tokenizer_dir}")
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
    
    logger.info(f"✓ Export complete! Files saved to {out_path}")


def upload_to_hub(output_dir: str, repo_id: str, private: bool = False) -> None:
    """Upload exported model to HuggingFace Hub."""
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )
    
    logger.info(f"{'='*60}")
    logger.info(f"Uploading to HuggingFace Hub: {repo_id}")
    logger.info(f"Repository visibility: {'Private' if private else 'Public'}")
    logger.info(f"{'='*60}")
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        logger.info("Creating/checking repository...")
        create_repo(repo_id, private=private, exist_ok=True)
        logger.info(f"✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise
    
    # Upload all files
    out_path = Path(output_dir)
    files_to_upload = list(out_path.iterdir())
    
    logger.info(f"Uploading {len(files_to_upload)} files...")
    for idx, file_path in enumerate(files_to_upload, 1):
        if file_path.is_file():
            file_size = file_path.stat().st_size / (1024**3)  # Size in GB
            logger.info(f"  [{idx}/{len(files_to_upload)}] {file_path.name} ({file_size:.2f} GB)")
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    commit_message=f"Upload {file_path.name}",
                )
            except Exception as e:
                logger.error(f"Failed to upload {file_path.name}: {e}")
                raise
    
    logger.info(f"{'='*60}")
    logger.info(f"✓ Upload complete!")
    logger.info(f"View your model at: https://huggingface.co/{repo_id}")
    logger.info(f"{'='*60}")


def main() -> None:
    args = parse_args()
    
    # Validate upload arguments
    if args.upload and not args.repo_id:
        logger.error("--repo-id is required when --upload is set")
        return
    
    if args.upload and not HF_HUB_AVAILABLE:
        logger.error("huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        return

    try:
        config_dict = load_training_config(args.config)
        ensure_single_parallelism(config_dict)

        init_distributed()
        setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=1)

        model_cfg = build_model_config(config_dict)
        model = instantiate_model(model_cfg)
        load_checkpoint(model, args.checkpoint)
        hf_state = convert_state_dict_to_hf(model)
        save_hf_artifacts(model_cfg, hf_state, args.output_dir, args.tokenizer_dir, args.dtype)
        
        # Upload to HuggingFace Hub if requested
        if args.upload:
            upload_to_hub(args.output_dir, args.repo_id, args.private)
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()