"""Standalone CODI-style inference utilities.

Implements latent-vector iteration (CODI) for a generic HuggingFace-style
causal LM, plus helpers to install latent vectors as special token embeddings
and sanity-check equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class CodiSanityCheck:
    """Result of checking token-embedding equivalence to CODI latents."""

    ok: bool
    max_abs_diff: float
    per_token_max_abs_diff: list[float]
    token_strings: list[str]


def _get_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    raise ValueError("Could not locate input embedding layer on model")


def _stack_latents(latent_vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    stacked = []
    for vec in latent_vectors:
        if vec.dim() == 3:
            vec = vec.squeeze(0).squeeze(0)
        elif vec.dim() == 2:
            vec = vec.squeeze(0)
        stacked.append(vec)
    return torch.stack(stacked, dim=0)


def codi_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
    max_new_tokens: int = 256,
    num_latent_iterations: int = 6,
    temperature: float = 0.1,
    top_k: int = 40,
    top_p: float = 0.95,
    greedy: bool = True,
    return_latent_vectors: bool = False,
    remove_eos: bool = False,
    output_hidden_states: bool = False,
    skip_thinking: bool = False,
    sot_token_id: Optional[int] = None,
    eot_token_id: Optional[int] = None,
    projection: Optional[torch.nn.Module] = None,
) -> dict:
    """Run CODI-style latent inference with a generic causal LM.

    Returns a dict with keys: sequences, latent_vectors, latent_vectors_post_prj,
    hidden_states (when requested).
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    device = input_ids.device
    batch_size = input_ids.shape[0]

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # Add start/end-of-thought tokens (mirror CODI.generate behavior)
    if remove_eos:
        if sot_token_id is None:
            raise ValueError("sot_token_id must be provided when remove_eos=True")
        bot_tensor = torch.tensor([sot_token_id], dtype=torch.long, device=device).expand(
            batch_size, 1
        )
    else:
        if skip_thinking:
            if eot_token_id is None:
                raise ValueError("eot_token_id must be provided when skip_thinking=True")
            bot_tensor = torch.tensor(
                [tokenizer.eos_token_id, eot_token_id],
                dtype=torch.long,
                device=device,
            ).expand(batch_size, 2)
        else:
            if sot_token_id is None:
                raise ValueError("sot_token_id must be provided when skip_thinking=False")
            bot_tensor = torch.tensor(
                [tokenizer.eos_token_id, sot_token_id],
                dtype=torch.long,
                device=device,
            ).expand(batch_size, 2)

    input_ids = torch.cat([input_ids, bot_tensor], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones(batch_size, bot_tensor.shape[1], device=device)], dim=1
    )

    latent_vectors = []
    latent_vectors_pre_prj = []
    latent_vectors_post_prj = []
    latent_inputs = []
    all_hidden_states = []

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values

        if output_hidden_states:
            prompt_hidden_states = outputs.hidden_states
            if prompt_hidden_states[0].shape[1] != input_ids.shape[1]:
                prompt_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                )
                if prompt_out.hidden_states is not None:
                    prompt_hidden_states = prompt_out.hidden_states
            hs = torch.stack([h for h in prompt_hidden_states], dim=0)
            all_hidden_states.append(hs)

        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        if return_latent_vectors:
            latent_vectors_pre_prj.append(latent_embd.clone())

        if projection is None and getattr(model, "use_prj", False) and hasattr(model, "prj"):
            projection = model.prj

        if projection is not None:
            target_dtype = getattr(model, "dtype", None) or next(model.parameters()).dtype
            latent_embd = projection(latent_embd).to(dtype=target_dtype)

        if return_latent_vectors:
            latent_vectors.append(latent_embd.clone())
            latent_vectors_post_prj.append(latent_embd.clone())

        if not skip_thinking:
            for _ in range(num_latent_iterations):
                if return_latent_vectors:
                    latent_inputs.append(latent_embd.clone())

                outputs = model(
                    inputs_embeds=latent_embd,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = outputs.past_key_values

                if output_hidden_states:
                    hs = torch.stack([h for h in outputs.hidden_states], dim=0)
                    all_hidden_states.append(hs)

                latent_embd = outputs.hidden_states[-1][:, -1:, :]

                if return_latent_vectors:
                    latent_vectors_pre_prj.append(latent_embd.clone())

                if projection is not None:
                    target_dtype = getattr(model, "dtype", None) or next(model.parameters()).dtype
                    latent_embd = projection(latent_embd).to(dtype=target_dtype)

                if return_latent_vectors:
                    latent_vectors.append(latent_embd.clone())
                    latent_vectors_post_prj.append(latent_embd.clone())

        if not skip_thinking and eot_token_id is None:
            raise ValueError("eot_token_id must be provided when skip_thinking=False")

        eot_ids = torch.tensor([eot_token_id], dtype=torch.long, device=device)
        eot_emb = _get_input_embeddings(model)(eot_ids).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        output = eot_emb

        generated = []
        for _ in range(max_new_tokens):
            if output is not None:
                out = model(
                    inputs_embeds=output,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=output_hidden_states,
                )
            else:
                out = outputs

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.config.vocab_size - 1]

            if greedy:
                next_token = logits.argmax(dim=-1)
            else:
                logits = logits / temperature
                if top_k > 1:
                    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                    min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                    logits[logits < min_top_k_value] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    if sorted_indices_to_remove.any():
                        sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                        sorted_indices_to_remove[:, 0] = False
                    for b in range(logits.size(0)):
                        logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0)

            generated.append(next_token.unsqueeze(1))

            if (next_token == tokenizer.eos_token_id).all():
                break

            output = _get_input_embeddings(model)(next_token).to(device).unsqueeze(1)

        if generated:
            sequences = torch.cat(generated, dim=1)
        else:
            sequences = torch.empty((batch_size, 0), dtype=torch.long, device=device)

    result = {"sequences": sequences}

    if return_latent_vectors:
        result["latent_vectors"] = latent_vectors
        result["latent_vectors_pre_prj"] = latent_vectors_pre_prj
        result["latent_vectors_post_prj"] = latent_vectors_post_prj
        result["latent_inputs"] = latent_inputs

    if output_hidden_states and all_hidden_states:
        result["hidden_states"] = torch.cat(all_hidden_states, dim=2)

    if remove_eos:
        pass

    return result


def add_latent_tokens(
    model,
    tokenizer,
    latent_vectors: Iterable[torch.Tensor],
    token_prefix: str = "v",
) -> list[str]:
    """Add <v1>..<vN> tokens and install embeddings from latent_vectors.

    Returns the list of token strings added.
    """
    latent_vectors = list(latent_vectors)
    tokens = [f"<{token_prefix}{i}>" for i, _ in enumerate(latent_vectors, start=1)]

    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in tokens if t not in existing]

    # assert len(to_add) == len(latent_vectors), f"Some tokens already exist in vocab: {set(tokens) - set(to_add)}; cannot safely add latent tokens"
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    try:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(len(tokenizer))

    embed = _get_input_embeddings(model)
    weight = embed.weight.data

    for token, vec in zip(tokens, latent_vectors):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Token {token} was not added to tokenizer")
        if vec.dim() > 1:
            vec = vec.squeeze()
        weight[token_id, :] = vec.to(device=weight.device, dtype=weight.dtype)

    return tokens


def codi_inference_with_tokens(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    num_latent_iterations: int = 6,
    temperature: float = 0.1,
    top_k: int = 40,
    top_p: float = 0.95,
    greedy: bool = True,
    output_hidden_states: bool = False,
    skip_thinking: bool = False,
    sot_token_id: Optional[int] = None,
    eot_token_id: Optional[int] = None,
    projection: Optional[torch.nn.Module] = None,
    token_prefix: str = "v",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[dict, list[str], CodiSanityCheck]:
    """Run CODI inference, add latent tokens, and sanity-check hidden states."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)

    result = codi_generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        num_latent_iterations=num_latent_iterations,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        return_latent_vectors=True,
        output_hidden_states=output_hidden_states,
        skip_thinking=skip_thinking,
        sot_token_id=sot_token_id,
        eot_token_id=eot_token_id,
        projection=projection,
    )

    latent_vectors = result.get("latent_vectors", [])
    latent_inputs = result.get("latent_inputs", [])
    projected = result.get("latent_vectors_post_prj", [])
    if latent_inputs:
        token_vectors = latent_inputs
    elif projected:
        token_vectors = projected
    else:
        token_vectors = latent_vectors

    token_strings = add_latent_tokens(
        model=model,
        tokenizer=tokenizer,
        latent_vectors=token_vectors,
        token_prefix=token_prefix,
    )

    expected_latents = latent_vectors
    if result.get("latent_vectors_pre_prj"):
        pre_prj = result["latent_vectors_pre_prj"]
        if len(pre_prj) > 1:
            expected_latents = pre_prj[1:]


    string_check, string_text = sanity_check_latent_tokens_tokenized_string(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        token_strings=token_strings,
        latent_vectors=expected_latents,
        sot_token_id=sot_token_id,
        eot_token_id=eot_token_id,
        atol=atol,
        rtol=rtol,
    )

    return result, token_strings, string_check, string_text


def sanity_check_latent_tokens_tokenized_string(
    model,
    tokenizer,
    prompt: str,
    token_strings: Iterable[str],
    latent_vectors: Iterable[torch.Tensor],
    sot_token_id: Optional[int] = None,
    eot_token_id: Optional[int] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> CodiSanityCheck:
    """Check that tokenizing the full string yields <v*> positions matching CODI latents."""
    model.eval()

    if sot_token_id is None or eot_token_id is None:
        raise ValueError("sot_token_id and eot_token_id must be provided for string sanity check")

    tokens = list(token_strings)
    sot_token = tokenizer.convert_ids_to_tokens(sot_token_id)
    eot_token = tokenizer.convert_ids_to_tokens(eot_token_id)
    eos_token = tokenizer.eos_token or ""

    # Ensure bot/eot tokens are treated as special tokens in string tokenization.
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [sot_token, eot_token]}
    )
    if added > 0:
        raise ValueError(
            "sot/eot tokens were not in the tokenizer vocab; cannot run string sanity check safely."
        )

    prompt_tokens = tokenizer.tokenize(prompt)
    full_text = tokenizer.convert_tokens_to_string(
        prompt_tokens + [eos_token, sot_token] + tokens + [eot_token]
    ).strip()

    device = next(model.parameters()).device
    v_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(t) for t in tokens],
        device=device,
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    input_ids = None
    attention_mask = None
    indices = None
    inputs = tokenizer(full_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer does not define eos_token_id")

    prefix_ids = torch.cat(
        [
            prompt_ids,
            torch.tensor([[eos_id, sot_token_id]], device=device),
        ],
        dim=1,
    )
    if input_ids.shape[1] < prefix_ids.shape[1] + len(tokens) + 1:
        raise ValueError("Tokenized input is shorter than expected prefix + v tokens + eot")
    if not torch.equal(input_ids[0, : prefix_ids.shape[1]], prefix_ids[0]):
        raise ValueError("Tokenized full string does not match prompt + eos + sot prefix")

    v_start = prefix_ids.shape[1]
    v_end = v_start + len(tokens)
    if not torch.equal(input_ids[0, v_start:v_end], v_ids):
        raise ValueError("Tokenized full string does not place <v*> tokens at expected positions")
    if input_ids[0, v_end].item() != eot_token_id:
        raise ValueError("Tokenized full string does not place eot token after <v*> tokens")

    indices = list(range(v_start, v_end))

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    final_hs = outputs.hidden_states[-1][:, indices, :].squeeze(0)
    latents = _stack_latents(latent_vectors).to(device=final_hs.device, dtype=final_hs.dtype)

    if final_hs.shape[0] != latents.shape[0]:
        raise ValueError(
            f"Latent length mismatch: hidden_states={final_hs.shape[0]} vs latents={latents.shape[0]}"
        )

    diffs = (final_hs - latents).abs().max(dim=1).values
    max_abs_diff = float(diffs.max().item()) if diffs.numel() else 0.0
    ok = torch.allclose(final_hs, latents, atol=atol, rtol=rtol)

    check = CodiSanityCheck(
        ok=bool(ok),
        max_abs_diff=max_abs_diff,
        per_token_max_abs_diff=[float(x.item()) for x in diffs],
        token_strings=tokens,
    )
    return check, full_text


def _decode_generated(tokenizer, sequences: torch.Tensor) -> str:
    return tokenizer.decode(sequences[0], skip_special_tokens=True).strip()


def _load_model_and_tokenizer(model_name_or_path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from circuit_tracer import ReplacementModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = ReplacementModel.from_pretrained(
        model_name_or_path,
        "mntss/transcoder-Llama-3.2-1B",
        backend="nnsight",
    )
    model.to(device)
    return model, tokenizer


def _build_prj_module(
    dim: int,
    prj_dim: int,
    prj_dropout: float,
    prj_no_ln: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    prj = torch.nn.Sequential(
        torch.nn.Dropout(prj_dropout),
        torch.nn.Linear(dim, prj_dim),
        torch.nn.GELU(),
        torch.nn.Linear(prj_dim, dim),
    )
    if not prj_no_ln:
        prj.add_module("ln", torch.nn.LayerNorm(dim))
    return prj.to(dtype=dtype, device=device)


def _load_prj_from_checkpoint(
    checkpoint_path: str, model: torch.nn.Module
) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu")
    prj_keys = [k for k in state.keys() if k.startswith("prj.")]
    if not prj_keys:
        raise ValueError(f"No prj.* keys found in checkpoint: {checkpoint_path}")

    dim = model.config.hidden_size
    prj_weight = state["prj.1.weight"]
    prj_dim = prj_weight.shape[0]
    prj_no_ln = not any(k.startswith("prj.ln.") for k in prj_keys)
    prj_dropout = 0.0

    dtype = getattr(model, "dtype", None) or next(model.parameters()).dtype
    device = next(model.parameters()).device
    prj = _build_prj_module(dim, prj_dim, prj_dropout, prj_no_ln, dtype, device)

    prj_state = {k.replace("prj.", ""): v for k, v in state.items() if k.startswith("prj.")}
    prj.load_state_dict(prj_state, strict=True)
    return prj


def _load_prj(
    prj_path: Optional[str],
    prj_source_checkpoint: Optional[str],
    model: torch.nn.Module,
) -> Optional[torch.nn.Module]:
    import os

    if prj_path is None and prj_source_checkpoint is None:
        return None

    if prj_path is not None:
        if os.path.exists(prj_path):
            payload = torch.load(prj_path, map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload:
                state_dict = payload["state_dict"]
                dim = payload["dim"]
                prj_dim = payload["prj_dim"]
                prj_no_ln = payload["prj_no_ln"]
                prj_dropout = payload.get("prj_dropout", 0.0)
                dtype = getattr(model, "dtype", None) or next(model.parameters()).dtype
                device = next(model.parameters()).device
                prj = _build_prj_module(dim, prj_dim, prj_dropout, prj_no_ln, dtype, device)
                prj.load_state_dict(state_dict, strict=True)
                return prj
            if isinstance(payload, dict):
                # Assume raw state_dict; infer shape from weights.
                dim = model.config.hidden_size
                prj_dim = payload["1.weight"].shape[0]
                prj_no_ln = not any(k.startswith("ln.") for k in payload.keys())
                dtype = getattr(model, "dtype", None) or next(model.parameters()).dtype
                device = next(model.parameters()).device
                prj = _build_prj_module(dim, prj_dim, 0.0, prj_no_ln, dtype, device)
                prj.load_state_dict(payload, strict=True)
                return prj
        if prj_source_checkpoint is None:
            raise FileNotFoundError(f"prj_path not found: {prj_path}")

    prj = _load_prj_from_checkpoint(prj_source_checkpoint, model)
    if prj_path is not None:
        torch.save(
            {
                "state_dict": prj.state_dict(),
                "dim": model.config.hidden_size,
                "prj_dim": prj[1].out_features,
                "prj_no_ln": not any(name == "ln" for name, _ in prj.named_children()),
                "prj_dropout": 0.0,
            },
            prj_path,
        )
    return prj


def _main():
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run CODI inference and latent-token sanity check.")
    parser.add_argument("model", help="Model name or path")
    # parser.add_argument("prompt", help="Prompt text")
    parser.add_argument("--num-latents", type=int, default=6, help="Number of latent iterations")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--device", default=None, help="Device (e.g., cuda, cpu). Defaults to cuda if available.")
    parser.add_argument(
        "--use-codi-wrapper",
        action="store_true",
        help="Load CODI.from_pretrained (neel-codi) instead of a plain HF model.",
    )
    parser.add_argument(
        "--prj-path",
        default=None,
        help="Path to a saved projection module (prj) state dict.",
    )
    parser.add_argument(
        "--prj-source-checkpoint",
        default=None,
        help="Path to a CODI checkpoint (pytorch_model.bin) containing prj.* keys.",
    )
    parser.add_argument(
        "--with-latent-tokens",
        action="store_true",
        help="Add latent tokens and run the sanity check (mutates token embeddings).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = _load_model_and_tokenizer(args.model, device)
    projection = _load_prj(args.prj_path, args.prj_source_checkpoint, model)


    prompts = [(3,2,4), (3, 2, 5), (6, 4, 3), (5, 1, 2), (4, 3, 4)]
    if args.with_latent_tokens:
        print("Mutates embeddings; only run a single prompt per process.")
        prompts = prompts[-1:]

    for X, Y, Z in prompts:
        prompt = f"""A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""
        result, token_strings, string_check, string_text = codi_inference_with_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            num_latent_iterations=args.num_latents,
            greedy=True,
            sot_token_id=tokenizer.convert_tokens_to_ids("<|bocot|>"),
            eot_token_id=tokenizer.convert_tokens_to_ids("<|eocot|>"),
            projection=projection,
        )

        decoded = _decode_generated(tokenizer, result["sequences"])

        print("Prompt:", prompt)
        print("Generated output:")
        print(decoded)
        if args.with_latent_tokens:
            print("")
            print("Latent tokens:", " ".join(token_strings))
            print("String sanity check ok:", string_check.ok)
            print("String max abs diff:", string_check.max_abs_diff)
            print("String sanity check text:")
            print(string_text)


if __name__ == "__main__":
    _main()
