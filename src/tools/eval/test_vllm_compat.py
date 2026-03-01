#!/usr/bin/env python
"""Test vLLM compatibility with exported HF checkpoint.

Compares picotron greedy generation with vLLM greedy generation.
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
import importlib.machinery


# Stub flash_attn
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
        bsz, seqlen, heads, hd = q.shape
        q = q.permute(0, 2, 1, 3).reshape(bsz * heads, seqlen, hd)
        k = k.permute(0, 2, 1, 3).reshape(bsz * heads, seqlen, hd)
        v = v.permute(0, 2, 1, 3).reshape(bsz * heads, seqlen, hd)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return out.reshape(bsz, heads, seqlen, hd).permute(0, 2, 1, 3)

    def _layer_norm_fn(hs, w, b, residual=None, eps=1e-5, is_rms_norm=False, **_):
        if residual is not None:
            hs = hs + residual
        x = hs.float()
        if is_rms_norm:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (w * x.to(w.dtype)).to(hs.dtype)

    interface.flash_attn_func = _flash_attn_func
    rotary.apply_rotary_emb = _apply_rotary_emb
    layer_norm.layer_norm_fn = _layer_norm_fn

    layers_ns = types.SimpleNamespace(rotary=rotary)
    triton_ns = types.SimpleNamespace(layer_norm=layer_norm)
    ops_ns = types.SimpleNamespace(triton=triton_ns)
    flash_attn_root.flash_attn_interface = interface
    flash_attn_root.layers = layers_ns
    flash_attn_root.ops = ops_ns

    for name, mod in [
        ("flash_attn", flash_attn_root),
        ("flash_attn.flash_attn_interface", interface),
        ("flash_attn.layers.rotary", rotary),
        ("flash_attn.ops.triton.layer_norm", layer_norm),
    ]:
        if isinstance(mod, types.ModuleType):
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = mod
    sys.modules["flash_attn.layers"] = layers_ns
    sys.modules["flash_attn.ops"] = ops_ns
    sys.modules["flash_attn.ops.triton"] = triton_ns


_ensure_stubs()

from transformers import AutoConfig
from picotron.model import Qwen3Model as PicotronQwen3
from picotron.process_group_manager import setup_process_group_manager


def init_dist():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29597")
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)
    setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=1)


@torch.no_grad()
def main():
    device = "cuda:0"
    config_path = "configs/main_runs_v2/qwen-3-200M-v3.json"
    ckpt_path = "models/main_runs_v2/qwen-3-200M-v3/20000/weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth"
    hf_model_path = "/tmp/export_200M_v3"

    init_dist()

    # Load picotron model
    print("Loading picotron model...")
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
    pico = PicotronQwen3(config=model_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    pico.load_state_dict(state, strict=False)
    pico = pico.to(device).eval()

    # Load vLLM
    print("Loading vLLM model...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=hf_model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        enforce_eager=True,
    )

    # Generate with picotron
    prompt = [2348, 105, 196, 0, 8, 0, 0, 2, 0, 0, 0]  # <bos> 180s 0inc 1800 2000
    max_tokens = 40
    pico_tokens = list(prompt)

    print("\nPicotron greedy generation:")
    for step in range(max_tokens):
        inp = torch.tensor([pico_tokens], device=device)
        logits = pico(inp)[:, -1, :]
        tok = logits.argmax(-1).item()
        if tok >= 2346:  # termination or special
            break
        pico_tokens.append(tok)
    pico_moves = [t for t in pico_tokens[len(prompt) :] if 378 <= t <= 2345]
    print(f"  Tokens: {pico_tokens[len(prompt) :]}")
    print(f"  Moves: {len(pico_moves)} moves generated")

    # Generate with vLLM
    print("\nvLLM greedy generation:")
    from vllm import TokensPrompt

    params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    results = llm.generate(TokensPrompt(prompt_token_ids=prompt), params)
    vllm_tokens = list(results[0].outputs[0].token_ids)
    vllm_moves = [t for t in vllm_tokens if 378 <= t <= 2345]
    print(f"  Tokens: {vllm_tokens}")
    print(f"  Moves: {len(vllm_moves)} moves generated")

    # Compare
    print("\nComparison:")
    match = 0
    for i, (p, v) in enumerate(zip(pico_tokens[len(prompt) :], vllm_tokens)):
        if p == v:
            match += 1
        else:
            print(f"  First mismatch at position {i}: pico={p}, vllm={v}")
            break
    else:
        print(f"  All {match} tokens match!")

    total = min(len(pico_tokens) - len(prompt), len(vllm_tokens))
    print(f"  Matching tokens: {match}/{total}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
