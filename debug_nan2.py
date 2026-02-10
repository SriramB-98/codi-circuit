#!/usr/bin/env python3
"""Test whether gradients cause the NaN issue."""
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
    model=base_model, tokenizer=tokenizer, prompt=PROMPT,
    max_new_tokens=256, num_latent_iterations=6, greedy=True,
    sot_token_id=tokenizer.convert_tokens_to_ids("<|bocot|>"),
    eot_token_id=tokenizer.convert_tokens_to_ids("<|eocot|>"),
    projection=_load_prj("./checkpoints/bcywinski/codi_llama1b-answer_only/prj.pt", None, base_model),
)
del result
attribution_prompt = string_text

model = ReplacementModel.from_pretrained(
    "meta-llama/Llama-3.2-1B", TRANSCODER_SET,
    backend="transformerlens", hf_model=base_model,
)
model.eval()

tokens = model.ensure_tokenized(attribution_prompt)
print(f"tokens shape: {tokens.shape}")

# Test 5: Forward pass WITH gradients (no hooks)
print("\n=== Test 5: Forward WITH gradients, no hooks ===")
residual = model.forward(tokens.expand(64, -1), stop_at_layer=model.cfg.n_layers)
has_nan = residual.isnan().any().item()
print(f"Residual has_nan: {has_nan}")
if has_nan:
    # Find first nan position
    nan_mask = residual.isnan().any(dim=-1)  # (batch, pos)
    print(f"NaN batch items: {nan_mask.any(dim=1).sum().item()}")
del residual

# Test 6: Setup attribution + forward with hooks (exactly like attribution code)
print("\n=== Test 6: setup_attribution + forward with hooks (like attribution) ===")
ctx = model.setup_attribution(tokens)
print(f"setup_attribution done, total_active_feats={ctx.activation_matrix._nnz()}")

batch_size = 64
with ctx.install_hooks(model):
    residual = model.forward(tokens.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
    print(f"Residual has_nan: {residual.isnan().any().item()}")

    # Check each cached resid activation
    for i, act in enumerate(ctx._resid_activations):
        if act is not None:
            has_nan = act.isnan().any().item()
            abs_max = act.abs().max().item() if not has_nan else 'NaN'
            print(f"  resid[{i}]: has_nan={has_nan}, abs_max={abs_max}, dtype={act.dtype}")

    ctx._resid_activations[-1] = model.ln_final(residual)
    print(f"After ln_final: has_nan={ctx._resid_activations[-1].isnan().any().item()}")

# Test 7: Forward with hooks but batch_size=1
print("\n=== Test 7: setup_attribution + forward with hooks (batch_size=1) ===")
ctx2 = model.setup_attribution(tokens)
with ctx2.install_hooks(model):
    residual2 = model.forward(tokens.unsqueeze(0), stop_at_layer=model.cfg.n_layers)
    print(f"Residual has_nan: {residual2.isnan().any().item()}")
    for i, act in enumerate(ctx2._resid_activations):
        if act is not None:
            has_nan = act.isnan().any().item()
            abs_max = act.abs().max().item() if not has_nan else 'NaN'
            print(f"  resid[{i}]: has_nan={has_nan}, abs_max={abs_max}")
