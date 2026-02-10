#!/usr/bin/env python3
"""Minimal script to test whether the replacement model forward pass produces NaN."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from circuit_tracer import ReplacementModel
from codi_inference import codi_inference_with_tokens, _load_prj

MODEL_PATH = "/workspace/codi-circuit/my_models/merged_bcywinski_codi_llama1b-answer_only"
TRANSCODER_SET = "mntss/transcoder-Llama-3.2-1B"
PROMPT = "A team starts with 3 members. 2 members leave the team. Then each remaining member recruits 4 additional people. How many people are there now on the team? Give the answer only and nothing else."

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Running CODI inference...")
result, token_strings, string_check, string_text = codi_inference_with_tokens(
    model=base_model,
    tokenizer=tokenizer,
    prompt=PROMPT,
    max_new_tokens=256,
    num_latent_iterations=6,
    greedy=True,
    sot_token_id=tokenizer.convert_tokens_to_ids("<|bocot|>"),
    eot_token_id=tokenizer.convert_tokens_to_ids("<|eocot|>"),
    projection=_load_prj("./checkpoints/bcywinski/codi_llama1b-answer_only/prj.pt", None, base_model),
)
del result
attribution_prompt = string_text
print(f"Attribution prompt: {attribution_prompt[:100]}...")

# Test 1: Direct forward pass with base model (no replacement)
print("\n=== Test 1: Base model forward pass ===")
input_ids = tokenizer(attribution_prompt, return_tensors="pt").input_ids
print(f"input_ids shape: {input_ids.shape}")
with torch.no_grad():
    outputs = base_model(input_ids.to(base_model.device))
    print(f"Base model logits has_nan: {outputs.logits.isnan().any().item()}")

# Test 2: Replacement model forward pass with batch_size=1
print("\n=== Test 2: Replacement model forward pass (batch_size=1) ===")
model = ReplacementModel.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    TRANSCODER_SET,
    backend="transformerlens",
    hf_model=base_model,
)
model.eval()

tokens = model.ensure_tokenized(attribution_prompt)
print(f"tokens shape: {tokens.shape}, n_tokens: {len(tokens)}")

# Run forward pass with batch_size=1
with torch.no_grad():
    residual = model.forward(tokens.unsqueeze(0), stop_at_layer=model.cfg.n_layers)
    print(f"Residual has_nan: {residual.isnan().any().item()}, abs_max: {residual.abs().max().item() if not residual.isnan().any().item() else 'NaN'}")

# Test 3: Replacement model forward pass with batch_size=64 (like the actual run)
print("\n=== Test 3: Replacement model forward pass (batch_size=64) ===")
with torch.no_grad():
    residual64 = model.forward(tokens.expand(64, -1), stop_at_layer=model.cfg.n_layers)
    print(f"Residual64 has_nan: {residual64.isnan().any().item()}, abs_max: {residual64.abs().max().item() if not residual64.isnan().any().item() else 'NaN'}")

# Test 4: Check layer by layer
print("\n=== Test 4: Layer-by-layer forward pass ===")
with torch.no_grad():
    for stop_layer in range(1, model.cfg.n_layers + 1):
        residual_l = model.forward(tokens.unsqueeze(0), stop_at_layer=stop_layer)
        has_nan = residual_l.isnan().any().item()
        abs_max = residual_l.abs().max().item() if not has_nan else 'NaN'
        print(f"  stop_at_layer={stop_layer}: has_nan={has_nan}, abs_max={abs_max}")
        if has_nan:
            break
