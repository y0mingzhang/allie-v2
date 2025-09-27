import os
import random
import sys
import types

import importlib.machinery

import torch
from transformers import LlamaConfig, LlamaForCausalLM


os.environ.setdefault("FLASH_ATTEN", "0")
os.environ.setdefault("DEVICE", "cpu")

flash_attn_stub = types.ModuleType("flash_attn")
flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
flash_attn_stub.flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
flash_attn_stub.flash_attn_interface.flash_attn_func = (
    lambda q, k, v, causal=True: torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
    ).transpose(1, 2)
)

rotary_module = types.ModuleType("flash_attn.layers.rotary")
rotary_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.layers.rotary", loader=None)
rotary_module.apply_rotary_emb = lambda x, cos, sin, interleaved=False: x

layer_norm_module = types.ModuleType("flash_attn.ops.triton.layer_norm")
layer_norm_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.ops.triton.layer_norm", loader=None)
layer_norm_module.layer_norm_fn = (
    lambda hidden_states, weight, bias=None, **kwargs: torch.nn.functional.layer_norm(
        hidden_states, hidden_states.shape[-1:], weight, bias, kwargs.get("eps", 1e-5)
    )
)

flash_attn_stub.layers = types.SimpleNamespace(rotary=rotary_module)
flash_attn_stub.ops = types.SimpleNamespace(triton=types.SimpleNamespace(layer_norm=layer_norm_module))

sys.modules.setdefault("flash_attn", flash_attn_stub)
sys.modules.setdefault("flash_attn.flash_attn_interface", flash_attn_stub.flash_attn_interface)
sys.modules.setdefault("flash_attn.layers.rotary", rotary_module)
sys.modules.setdefault("flash_attn.ops.triton.layer_norm", layer_norm_module)

from picotron.model import Llama as PicotronLlama
from picotron.export_to_hf import convert_state_dict_to_hf


def _set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _build_tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=144,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
    )


def test_llama_matches_hf():
    _set_seed()
    cfg = _build_tiny_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    picotron_model = PicotronLlama(cfg).eval()
    hf_model = LlamaForCausalLM(cfg).eval()

    with torch.no_grad():
        picotron_state = picotron_model.state_dict()
        hf_state = hf_model.state_dict()
        for name, tensor in picotron_state.items():
            if name in hf_state:
                hf_state[name].copy_(tensor)
        hf_model.load_state_dict(hf_state)

    converted_state = convert_state_dict_to_hf(picotron_model)
    with torch.no_grad():
        hf_model.load_state_dict(converted_state, strict=True)

    picotron_model.to(device)
    hf_model.to(device)

    batch_size, seq_len = 3, 10
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(batch_size, seq_len), device=device)

    with torch.no_grad():
        picotron_logits = picotron_model(input_ids).to(torch.float32).cpu()
        hf_result = hf_model(input_ids)
        if hasattr(hf_result, "logits"):
            hf_logits = hf_result.logits.to(torch.float32).cpu()
        else:
            hf_logits = hf_result[0].to(torch.float32).cpu()


    torch.testing.assert_close(
        picotron_logits,
        hf_logits,
        atol=1e-3,
        rtol=1e-3,
    )

