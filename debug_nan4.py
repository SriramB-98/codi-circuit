#!/usr/bin/env python3
"""Quick check on transcoder properties and skip connection."""
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

print(f"skip_transcoder: {model.skip_transcoder}")
print(f"transcoders type: {type(model.transcoders)}")
print(f"transcoders.skip_connection: {model.transcoders.skip_connection}")
print(f"feature_input_hook: {model.feature_input_hook}")
print(f"feature_output_hook: {model.feature_output_hook}")

tokens = model.ensure_tokenized(attribution_prompt)
batch_size = 64

# Now reproduce the EXACT sequence from _run_attribution with offload=disk
print("\n=== Reproducing exact _run_attribution sequence ===")
from circuit_tracer.utils.disk_offload import offload_modules
from circuit_tracer.utils.salient_logits import compute_salient_logits

ctx = model.setup_attribution(tokens)
activation_matrix = ctx.activation_matrix
print(f"Active features: {activation_matrix._nnz()}")

# Offload transcoders (Phase 0 end)
offload_handles = offload_modules(model.transcoders, "disk")
print("Transcoders offloaded")

# Phase 1: forward pass
with ctx.install_hooks(model):
    residual = model.forward(tokens.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
    ctx._resid_activations[-1] = model.ln_final(residual)
print("Phase 1 forward done")

# Check residuals
for i, act in enumerate(ctx._resid_activations):
    if act is not None:
        has_nan = act.isnan().any().item()
        abs_max = act.abs().max().item() if not has_nan else 'NaN'
        print(f"  resid[{i}]: has_nan={has_nan}, abs_max={abs_max}, requires_grad={act.requires_grad}")

# Offload MLP blocks (like Phase 1 end in _run_attribution)
offload_handles += offload_modules([block.mlp for block in model.blocks], "disk")
print("MLP blocks offloaded")

# Phase 2: build input vectors
feat_layers, feat_pos, _ = activation_matrix.indices()
n_layers, n_pos, _ = activation_matrix.shape

logit_idx, logit_p, logit_vecs = compute_salient_logits(
    ctx.logits[0, -1], model.unembed.W_U,
    max_n_logits=2, desired_logit_prob=0.95,
)
print(f"Logits selected: {len(logit_idx)}")

# Offload embed/unembed
offload_handles += offload_modules([model.unembed, model.embed], "disk")
print("Embed/unembed offloaded")

# Phase 3: Logit attribution
total_active_feats = activation_matrix._nnz()
logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
n_logits = len(logit_idx)
total_nodes = logit_offset + n_logits
max_feature_nodes = 500

edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)

print(f"\nPhase 3: Computing logit attributions")
for i in range(0, len(logit_idx), batch_size):
    batch = logit_vecs[i : i + batch_size]
    rows = ctx.compute_batch(
        layers=torch.full((batch.shape[0],), n_layers),
        positions=torch.full((batch.shape[0],), n_pos - 1),
        inject_values=batch,
    )
    rows_cpu = rows.cpu()
    print(f"  Logit batch {i}: has_nan={rows_cpu.isnan().any().item()}, "
          f"nan_count={rows_cpu.isnan().sum().item()}, "
          f"abs_max={rows_cpu.abs().max().item() if not rows_cpu.isnan().any().item() else 'NaN'}")
    edge_matrix[i : i + batch.shape[0], :logit_offset] = rows_cpu
    row_to_node_index[i : i + batch.shape[0]] = torch.arange(i, i + batch.shape[0]) + logit_offset

print(f"\nedge_matrix has_nan: {edge_matrix.isnan().any().item()}")
print(f"edge_matrix nan_count: {edge_matrix.isnan().sum().item()}")

# Cleanup
for h in offload_handles:
    h()
