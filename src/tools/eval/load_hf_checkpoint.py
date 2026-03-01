"""Utility to load HuggingFace safetensors checkpoint into picotron Qwen3Model."""

import os
import torch
from safetensors.torch import load_file
from transformers import AutoConfig


def load_hf_qwen3(model_dir, device="cpu", dtype=torch.bfloat16):
    """Load a Qwen3 model from HF safetensors format into picotron Qwen3Model."""
    os.environ["FLASH_ATTEN"] = "0"
    from picotron.model import Qwen3Model

    config = AutoConfig.from_pretrained(model_dir)
    config.vocab_size = 2350
    config.max_position_embeddings = 1024

    model = Qwen3Model(config)

    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    mapped = {}
    for k, v in state_dict.items():
        new_k = (
            k.replace("model.embed_tokens", "embedding")
            .replace("model.layers", "decoder_layers")
            .replace("self_attn", "attention")
            .replace("model.norm", "final_norm")
            .replace("lm_head", "final_proj")
            .replace("o_proj", "out_proj")
        )
        mapped[new_k] = v

    result = model.load_state_dict(mapped, strict=False)
    if result.missing_keys:
        print(f"Warning: {len(result.missing_keys)} missing keys")
    if result.unexpected_keys:
        print(f"Warning: {len(result.unexpected_keys)} unexpected keys")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config
