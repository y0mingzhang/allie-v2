#!/usr/bin/env python
"""Test vLLM inference with exported HF checkpoint."""

from vllm import LLM, SamplingParams, TokensPrompt

model_path = "/tmp/export_200M_v3"

print("Loading vLLM model...")
llm = LLM(
    model=model_path,
    dtype="bfloat16",
    gpu_memory_utilization=0.3,
    max_model_len=1024,
    enforce_eager=True,
)

# <bos> <seconds:180> <increment:0> <elo:1800> <elo:2000>
prompt = [2348, 105, 196, 0, 8, 0, 0, 2, 0, 0, 0]

# Expected from picotron (from debug_vllm_compat.py autoregressive test):
expected = [
    1409,
    1584,
    1907,
    816,
    1635,
    2123,
    1138,
    1873,
    1382,
    1627,
    872,
    1315,
    1174,
    1507,
    916,
    970,
    1209,
    1018,
    1711,
    1283,
    600,
    1090,
    951,
    1993,
    1110,
    755,
    848,
    1202,
    637,
    1806,
]

params = SamplingParams(temperature=0, max_tokens=40, ignore_eos=True)
results = llm.generate(TokensPrompt(prompt_token_ids=prompt), params)
vllm_tokens = list(results[0].outputs[0].token_ids)

print(f"\nvLLM tokens:     {vllm_tokens[:30]}")
print(f"Expected tokens: {expected}")

match = 0
for i, (v, e) in enumerate(zip(vllm_tokens, expected)):
    if v == e:
        match += 1
    else:
        print(f"\nFirst mismatch at position {i}: vllm={v}, expected={e}")
        break
else:
    print(f"\nAll {match} tokens match!")

print(f"Matching: {match}/{min(len(vllm_tokens), len(expected))}")
