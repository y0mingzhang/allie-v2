#!/usr/bin/env python
"""Debug vLLM/HF compatibility by comparing picotron vs HF transformers outputs.

Isolates the exact point of divergence: RoPE, QK-norm, attention, or layernorm.
"""

import json
import os
import sys
import types

os.environ["DEVICE"] = "cuda"
os.environ["FLASH_ATTEN"] = "0"
os.environ["CONTEXT_PARALLEL"] = "0"
os.environ["DTYPE"] = "bfloat16"

import torch
import torch.distributed as dist
import torch.nn.functional as F


# Stub flash_attn for picotron import
def _ensure_stubs():
    flash_attn_root = types.ModuleType("flash_attn")
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    rotary = types.ModuleType("flash_attn.layers.rotary")
    layer_norm = types.ModuleType("flash_attn.ops.triton.layer_norm")

    def _rotate_half(x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_emb(x, cos, sin, interleaved=False):
        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return (x * cos) + (_rotate_half(x) * sin)

    def _flash_attn_func(q, k, v, causal=True, **_):
        bsz, seqlen, heads, head_dim = q.shape
        q_flat = q.permute(0, 2, 1, 3).reshape(bsz * heads, seqlen, head_dim)
        k_flat = k.permute(0, 2, 1, 3).reshape(bsz * heads, seqlen, head_dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(bsz * heads, seqlen, head_dim)
        out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, is_causal=causal)
        return out.reshape(bsz, heads, seqlen, head_dim).permute(0, 2, 1, 3)

    def _layer_norm_fn(
        hidden_states, weight, bias, residual=None, eps=1e-5, prenorm=False, is_rms_norm=False, **_
    ):
        if residual is not None:
            hidden_states = hidden_states + residual
        working = hidden_states.float()
        if is_rms_norm:
            variance = working.pow(2).mean(dim=-1, keepdim=True)
            working = working * torch.rsqrt(variance + eps)
        out = (weight * working.to(weight.dtype)).to(hidden_states.dtype)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out

    interface.flash_attn_func = _flash_attn_func
    rotary.apply_rotary_emb = _apply_rotary_emb
    layer_norm.layer_norm_fn = _layer_norm_fn

    layers_ns = types.SimpleNamespace(rotary=rotary)
    triton_ns = types.SimpleNamespace(layer_norm=layer_norm)
    ops_ns = types.SimpleNamespace(triton=triton_ns)
    flash_attn_root.flash_attn_interface = interface
    flash_attn_root.layers = layers_ns
    flash_attn_root.ops = ops_ns

    # Set __spec__ to avoid importlib.util.find_spec crashes
    import importlib

    for name, mod in [
        ("flash_attn", flash_attn_root),
        ("flash_attn.flash_attn_interface", interface),
        ("flash_attn.layers.rotary", rotary),
        ("flash_attn.ops.triton.layer_norm", layer_norm),
    ]:
        if isinstance(mod, types.ModuleType):
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = mod
    # Non-module namespaces
    sys.modules["flash_attn.layers"] = layers_ns
    sys.modules["flash_attn.ops"] = ops_ns
    sys.modules["flash_attn.ops.triton"] = triton_ns


_ensure_stubs()

from transformers import AutoConfig, AutoModelForCausalLM
from picotron.model import Qwen3Model as PicotronQwen3
from picotron.model import get_cos_sin, apply_rotary_pos_emb, TritonRMSNorm
from picotron.process_group_manager import setup_process_group_manager


def init_dist():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)
    setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=1)


def load_picotron_model(config_path, ckpt_path, device="cuda:0"):
    with open(config_path) as f:
        cfg = json.load(f)

    model_cfg = AutoConfig.from_pretrained(cfg["model"]["name"])
    for key in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    ]:
        if key in cfg["model"]:
            setattr(model_cfg, key, cfg["model"][key])

    model = PicotronQwen3(config=model_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, model_cfg


def export_and_load_hf(picotron_model, model_cfg, device="cuda:0"):
    """Export picotron weights to HF format in memory and load."""
    from collections import OrderedDict

    KEY_MAP = {
        "decoder_layers.": "model.layers.",
        "embedding.": "model.embed_tokens.",
        "final_norm.": "model.norm.",
        "final_proj.": "lm_head.",
        "attention.": "self_attn.",
        "out_proj": "o_proj",
    }

    state = picotron_model.state_dict()
    hf_state = OrderedDict()
    for key, tensor in state.items():
        hf_key = key
        for old, new in KEY_MAP.items():
            hf_key = hf_key.replace(old, new)
        hf_state[hf_key] = tensor.cpu()

    # If no lm_head, model uses tied embeddings
    has_lm_head = "lm_head.weight" in hf_state
    model_cfg.tie_word_embeddings = not has_lm_head
    if not has_lm_head:
        print("  (Model uses tied embeddings)")
    hf_model = AutoModelForCausalLM.from_config(model_cfg)
    missing, unexpected = hf_model.load_state_dict(hf_state, strict=False)
    if missing:
        print(f"HF missing keys: {missing}")
    if unexpected:
        print(f"HF unexpected keys: {unexpected}")
    hf_model = hf_model.to(device).eval()
    return hf_model


@torch.no_grad()
def compare_rope(model_cfg, device="cuda:0"):
    """Compare RoPE computation between picotron and HF transformers."""
    print("=" * 60)
    print("TEST 1: RoPE frequency computation")
    print("=" * 60)

    head_dim = getattr(model_cfg, "head_dim", 128)
    rope_theta = getattr(model_cfg, "rope_theta", 500000.0)
    seq_len = 32

    # Picotron RoPE
    pico_cos, pico_sin = get_cos_sin(seq_len, head_dim, base=rope_theta)
    pico_cos = pico_cos.to(device)
    pico_sin = pico_sin.to(device)

    # HF RoPE (from transformers Qwen3 implementation)
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
    )
    position = torch.arange(seq_len).unsqueeze(1).float()
    freqs = position * inv_freq.unsqueeze(0)
    hf_cos = torch.cos(freqs).to(torch.bfloat16).to(device)
    hf_sin = torch.sin(freqs).to(torch.bfloat16).to(device)

    # Picotron stores as [seq, head_dim] with repeat(1,2)
    # HF stores as [seq, head_dim//2]
    pico_cos_half = pico_cos[:, : head_dim // 2]
    pico_sin_half = pico_sin[:, : head_dim // 2]

    diff_cos = (pico_cos_half.float() - hf_cos.float()).abs().max().item()
    diff_sin = (pico_sin_half.float() - hf_sin.float()).abs().max().item()
    print(f"  Picotron cos shape: {pico_cos.shape}, HF cos shape: {hf_cos.shape}")
    print(f"  Max diff cos: {diff_cos:.2e}")
    print(f"  Max diff sin: {diff_sin:.2e}")
    print(f"  Match: {'YES' if diff_cos < 1e-4 and diff_sin < 1e-4 else 'NO'}")
    return pico_cos, pico_sin


@torch.no_grad()
def compare_qknorm(pico_model, hf_model, device="cuda:0"):
    """Compare QK-norm weights."""
    print("\n" + "=" * 60)
    print("TEST 2: QK-norm weights")
    print("=" * 60)

    for layer_idx in range(min(3, len(pico_model.decoder_layers))):
        pico_layer = pico_model.decoder_layers[layer_idx]
        hf_layer = hf_model.model.layers[layer_idx]

        pico_qn = pico_layer.attention.q_norm.weight
        hf_qn = hf_layer.self_attn.q_norm.weight.to(device)

        pico_kn = pico_layer.attention.k_norm.weight
        hf_kn = hf_layer.self_attn.k_norm.weight.to(device)

        diff_q = (pico_qn.float() - hf_qn.float()).abs().max().item()
        diff_k = (pico_kn.float() - hf_kn.float()).abs().max().item()
        print(f"  Layer {layer_idx}: q_norm diff={diff_q:.2e}, k_norm diff={diff_k:.2e}")


@torch.no_grad()
def compare_full_forward(pico_model, hf_model, model_cfg, device="cuda:0"):
    """Compare full forward pass outputs."""
    print("\n" + "=" * 60)
    print("TEST 3: Full forward pass comparison")
    print("=" * 60)

    input_ids = torch.tensor(
        [[2348, 105, 196, 0, 8, 0, 0, 2, 0, 0, 0, 1424, 940, 1240, 510]], device=device
    )
    seq_len = input_ids.shape[1]

    # Picotron forward
    pico_logits = pico_model(input_ids)
    pico_probs = torch.softmax(pico_logits[:, -1, :].float(), dim=-1)

    # HF forward
    hf_out = hf_model(input_ids)
    hf_logits = hf_out.logits
    hf_probs = torch.softmax(hf_logits[:, -1, :].float(), dim=-1)

    # Compare
    logit_diff = (pico_logits.float() - hf_logits.float()).abs()
    print(f"  Logit max diff: {logit_diff.max().item():.4e}")
    print(f"  Logit mean diff: {logit_diff.mean().item():.4e}")

    pico_top5 = pico_probs[0].topk(5)
    hf_top5 = hf_probs[0].topk(5)
    print(
        f"\n  Picotron top-5: {list(zip(pico_top5.indices.tolist(), [f'{p:.4f}' for p in pico_top5.values.tolist()]))}"
    )
    print(
        f"  HF top-5:       {list(zip(hf_top5.indices.tolist(), [f'{p:.4f}' for p in hf_top5.values.tolist()]))}"
    )

    if pico_top5.indices[0] == hf_top5.indices[0]:
        print("  Top-1 prediction: MATCH")
    else:
        print(
            f"  Top-1 prediction: MISMATCH (pico={pico_top5.indices[0].item()}, hf={hf_top5.indices[0].item()})"
        )

    return pico_logits, hf_logits


@torch.no_grad()
def compare_layer_by_layer(pico_model, hf_model, model_cfg, device="cuda:0"):
    """Hook into each layer and compare intermediate states."""
    print("\n" + "=" * 60)
    print("TEST 4: Layer-by-layer comparison")
    print("=" * 60)

    input_ids = torch.tensor(
        [[2348, 105, 196, 0, 8, 0, 0, 2, 0, 0, 0, 1424, 940, 1240, 510]], device=device
    )

    # Get embeddings
    pico_emb = pico_model.embedding(input_ids)
    hf_emb = hf_model.model.embed_tokens(input_ids)
    diff = (pico_emb.float() - hf_emb.float()).abs().max().item()
    print(f"  Embedding diff: {diff:.2e}")

    head_dim = getattr(model_cfg, "head_dim", 128)
    rope_theta = getattr(model_cfg, "rope_theta", 500000.0)
    seq_len = input_ids.shape[1]

    # Get picotron cos/sin
    pico_cos, pico_sin = get_cos_sin(seq_len, head_dim, base=rope_theta)
    pico_cos = pico_cos.to(device)
    pico_sin = pico_sin.to(device)

    # Run through picotron layers manually
    pico_x = pico_emb
    for i, layer in enumerate(pico_model.decoder_layers):
        # Picotron: norm -> attn -> residual -> norm -> mlp -> residual
        residual = pico_x
        pico_x_norm = layer.input_layernorm(pico_x)
        pico_x_attn = layer.attention(pico_x_norm, pico_cos, pico_sin)
        pico_x = residual + pico_x_attn
        residual = pico_x
        pico_x_norm2 = layer.post_attention_layernorm(pico_x)
        pico_x_mlp = layer.mlp(pico_x_norm2)
        pico_x = residual + pico_x_mlp

        if i < 3 or i == len(pico_model.decoder_layers) - 1:
            # Compare with HF: capture HF layer output via hook
            pass  # We'll compare after full pass

    # Run HF model and capture hidden states
    hf_out = hf_model(input_ids, output_hidden_states=True)
    for i in range(min(4, len(hf_out.hidden_states) - 1)):
        idx = i if i < 3 else len(hf_out.hidden_states) - 2
        # hidden_states[0] = embeddings, [1] = after layer 0, etc.
        # But picotron manually computes - let me just compare final output
        pass

    # Compare final hidden state before lm_head
    pico_final = pico_model.final_norm(pico_x)
    hf_final = hf_out.hidden_states[-1]  # Already normed? No, HF returns pre-norm
    # Actually HF model.layers output is post-layer, then model.norm is applied

    # Let's just compare the hidden states from HF
    for i, hs in enumerate(hf_out.hidden_states[:4]):
        print(
            f"  HF hidden_state[{i}] shape: {hs.shape}, "
            f"mean: {hs.float().mean():.4e}, std: {hs.float().std():.4e}"
        )

    # Final comparison
    diff = (pico_final.float() - hf_final.float()).abs()
    print(
        f"\n  Final hidden state diff: max={diff.max().item():.4e}, mean={diff.mean().item():.4e}"
    )


@torch.no_grad()
def compare_autoregressive(pico_model, hf_model, model_cfg, device="cuda:0", max_tokens=30):
    """Compare autoregressive generation token by token."""
    print("\n" + "=" * 60)
    print("TEST 5: Autoregressive generation")
    print("=" * 60)

    # Initial prompt: <bos> <seconds:180> <increment:0> <elo:1800> <elo:2000>
    prompt = [2348, 105, 196, 0, 8, 0, 0, 2, 0, 0, 0]
    input_ids = torch.tensor([prompt], device=device)

    head_dim = getattr(model_cfg, "head_dim", 128)
    rope_theta = getattr(model_cfg, "rope_theta", 500000.0)
    max_seq = len(prompt) + max_tokens

    pico_cos, pico_sin = get_cos_sin(max_seq, head_dim, base=rope_theta)
    pico_cos = pico_cos.to(device)
    pico_sin = pico_sin.to(device)

    pico_tokens = list(prompt)
    hf_tokens = list(prompt)

    for step in range(max_tokens):
        pico_input = torch.tensor([pico_tokens], device=device)
        hf_input = torch.tensor([hf_tokens], device=device)

        pico_logits = pico_model(pico_input)[:, -1, :]
        hf_logits = hf_model(hf_input).logits[:, -1, :]

        pico_tok = pico_logits.argmax(-1).item()
        hf_tok = hf_logits.argmax(-1).item()

        logit_diff = (pico_logits.float() - hf_logits.float()).abs().max().item()

        match = "OK" if pico_tok == hf_tok else "MISMATCH"
        print(
            f"  Step {step:2d}: pico={pico_tok:5d} hf={hf_tok:5d} "
            f"logit_diff={logit_diff:.4e} [{match}]"
        )

        if pico_tok != hf_tok:
            # Show top-5 for both
            pico_top5 = pico_logits[0].topk(5)
            hf_top5 = hf_logits[0].topk(5)
            print(f"         pico top5: {pico_top5.indices.tolist()}")
            print(f"         hf   top5: {hf_top5.indices.tolist()}")
            break

        pico_tokens.append(pico_tok)
        hf_tokens.append(hf_tok)


def main():
    device = "cuda:0"
    config_path = "configs/main_runs_v2/qwen-3-200M-engine-mix.json"
    ckpt_path = "models/main_runs_v2/qwen-3-200M-engine-mix/10000/weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth"

    init_dist()

    print("Loading picotron model...")
    pico_model, model_cfg = load_picotron_model(config_path, ckpt_path, device)
    print(
        f"Model config: hidden={model_cfg.hidden_size}, layers={model_cfg.num_hidden_layers}, "
        f"heads={model_cfg.num_attention_heads}, head_dim={getattr(model_cfg, 'head_dim', 128)}, "
        f"rope_theta={getattr(model_cfg, 'rope_theta', 'default')}"
    )

    print("\nExporting to HF format and loading...")
    hf_model = export_and_load_hf(pico_model, model_cfg, device)

    compare_rope(model_cfg, device)
    compare_qknorm(pico_model, hf_model, device)
    compare_full_forward(pico_model, hf_model, model_cfg, device)
    compare_layer_by_layer(pico_model, hf_model, model_cfg, device)
    compare_autoregressive(pico_model, hf_model, model_cfg, device)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
