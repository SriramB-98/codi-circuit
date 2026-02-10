#!/usr/bin/env python3
"""Test whether bfloat16 conversion causes NaN."""
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

# THIS IS THE KEY LINE from visualize_llama_circuits.py line 193
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"\nConverting base_model to {dtype}...")
base_model.to(dtype)

model = ReplacementModel.from_pretrained(
    "meta-llama/Llama-3.2-1B", TRANSCODER_SET,
    backend="transformerlens", hf_model=base_model,
)
model.eval()

tokens = model.ensure_tokenized(attribution_prompt)
batch_size = 64

# Check model dtype
print(f"Model cfg dtype: {model.cfg.dtype}")
first_param = next(model.parameters())
print(f"First param dtype: {first_param.dtype}")

# Test: Forward with hooks (like attribution)
print("\n=== Test: Forward with hooks after bfloat16 conversion ===")
ctx = model.setup_attribution(tokens)
with ctx.install_hooks(model):
    residual = model.forward(tokens.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
    print(f"Residual has_nan: {residual.isnan().any().item()}")
    for i, act in enumerate(ctx._resid_activations):
        if act is not None:
            has_nan = act.isnan().any().item()
            abs_max = act.abs().max().item() if not has_nan else 'NaN'
            print(f"  resid[{i}]: has_nan={has_nan}, abs_max={abs_max}, dtype={act.dtype}")
            if has_nan:
                break
