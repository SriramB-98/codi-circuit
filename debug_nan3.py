#!/usr/bin/env python3
"""Test whether offloading transcoders before forward pass causes NaN."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from circuit_tracer import ReplacementModel
from circuit_tracer.utils.disk_offload import offload_modules
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
batch_size = 64

# Test A: Offload transcoders BEFORE forward (like current _run_attribution)
print("\n=== Test A: Offload transcoders BEFORE forward (reproducing bug) ===")
ctx_a = model.setup_attribution(tokens)
print(f"setup_attribution done, active_feats={ctx_a.activation_matrix._nnz()}")

# Offload transcoders (same as line 130 in _run_attribution)
reload_handles = offload_modules(model.transcoders, "disk")
print("Transcoders offloaded to disk")

with ctx_a.install_hooks(model):
    residual_a = model.forward(tokens.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
    print(f"Residual has_nan: {residual_a.isnan().any().item()}")
    for i, act in enumerate(ctx_a._resid_activations):
        if act is not None:
            has_nan = act.isnan().any().item()
            abs_max = act.abs().max().item() if not has_nan else 'NaN'
            print(f"  resid[{i}]: has_nan={has_nan}, abs_max={abs_max}")

# Reload transcoders for Test B
for h in reload_handles:
    h()
print("\nTranscoders reloaded")

# Test B: Offload transcoders AFTER forward (proposed fix)
print("\n=== Test B: Offload transcoders AFTER forward (proposed fix) ===")
ctx_b = model.setup_attribution(tokens)

with ctx_b.install_hooks(model):
    residual_b = model.forward(tokens.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
    print(f"Residual has_nan: {residual_b.isnan().any().item()}")
    for i, act in enumerate(ctx_b._resid_activations):
        if act is not None:
            has_nan = act.isnan().any().item()
            abs_max = act.abs().max().item() if not has_nan else 'NaN'
            print(f"  resid[{i}]: has_nan={has_nan}, abs_max={abs_max}")

# Now offload after forward
reload_handles_b = offload_modules(model.transcoders, "disk")
print("Transcoders offloaded after forward - no NaN in residuals!")

for h in reload_handles_b:
    h()
