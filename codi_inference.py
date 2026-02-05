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

    # Add start-of-thought token if provided
    if sot_token_id is not None:
        sot_tensor = torch.tensor([[tokenizer.eos_token_id, sot_token_id]], device=device)
        sot_tensor = sot_tensor.expand(batch_size, -1)
        input_ids = torch.cat([input_ids, sot_tensor], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones(batch_size, 2, device=device)], dim=1
            )

    latent_vectors = []
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
        past_length = input_ids.shape[1]

        if output_hidden_states:
            hs = torch.stack([h for h in outputs.hidden_states], dim=0)
            all_hidden_states.append(hs)

        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        if not skip_thinking:
            for _ in range(num_latent_iterations):
                if projection is not None:
                    latent_input = projection(latent_embd)
                    latent_input = latent_input.to(dtype=outputs.hidden_states[-1].dtype)
                else:
                    latent_input = latent_embd

                if return_latent_vectors:
                    latent_inputs.append(latent_input.clone())

                outputs = model(
                    inputs_embeds=latent_input,
                    past_key_values=past_key_values,
                    position_ids=torch.full(
                        (batch_size, 1), past_length, device=device, dtype=torch.long
                    ),
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = outputs.past_key_values
                past_length += 1

                if output_hidden_states:
                    hs = torch.stack([h for h in outputs.hidden_states], dim=0)
                    all_hidden_states.append(hs)

                latent_embd = outputs.hidden_states[-1][:, -1:, :]

                if return_latent_vectors:
                    latent_vectors.append(latent_embd.clone())
                    if projection is not None:
                        latent_vectors_post_prj.append(
                            projection(latent_embd).to(dtype=outputs.hidden_states[-1].dtype).clone()
                        )

        if eot_token_id is not None:
            eot_tensor = torch.tensor([[eot_token_id]], device=device)
            eot_tensor = eot_tensor.expand(batch_size, -1)

            outputs = model(
                input_ids=eot_tensor,
                past_key_values=past_key_values,
                position_ids=torch.full(
                    (batch_size, 1), past_length, device=device, dtype=torch.long
                ),
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values
            past_length += 1

        generated = []
        for _ in range(max_new_tokens):
            logits = outputs.logits[:, -1, :]

            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)

            if (next_token == tokenizer.eos_token_id).all():
                break

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                position_ids=torch.full(
                    (batch_size, 1), past_length, device=device, dtype=torch.long
                ),
                use_cache=True,
                output_hidden_states=output_hidden_states,
            )
            past_key_values = outputs.past_key_values
            past_length += 1

        if generated:
            generated_ids = torch.cat(generated, dim=1)
            sequences = torch.cat([input_ids, generated_ids], dim=1)
        else:
            sequences = input_ids

    result = {"sequences": sequences}

    if return_latent_vectors:
        result["latent_vectors"] = latent_vectors
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

    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
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

    check = sanity_check_latent_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        latent_vectors=latent_vectors,
        token_strings=token_strings,
        atol=atol,
        rtol=rtol,
    )

    return result, token_strings, check


def sanity_check_latent_tokens(
    model,
    tokenizer,
    prompt: str,
    latent_vectors: Iterable[torch.Tensor],
    token_strings: Iterable[str],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> CodiSanityCheck:
    """Compare final hidden states at <v*> positions vs CODI latent vectors."""
    model.eval()

    tokens = list(token_strings)
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"].to(next(model.parameters()).device)

    token_ids = torch.tensor(
        [[tokenizer.convert_tokens_to_ids(t) for t in tokens]],
        device=prompt_ids.device,
    )
    input_ids = torch.cat([prompt_ids, token_ids], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    final_hs = outputs.hidden_states[-1][:, -len(tokens) :, :].squeeze(0)
    latents = _stack_latents(latent_vectors).to(device=final_hs.device, dtype=final_hs.dtype)

    diffs = (final_hs - latents).abs().max(dim=1).values
    max_abs_diff = float(diffs.max().item()) if diffs.numel() else 0.0
    ok = torch.allclose(final_hs, latents, atol=atol, rtol=rtol)

    return CodiSanityCheck(
        ok=bool(ok),
        max_abs_diff=max_abs_diff,
        per_token_max_abs_diff=[float(x.item()) for x in diffs],
        token_strings=tokens,
    )


def _decode_generated(tokenizer, sequences: torch.Tensor, prompt_len: int) -> str:
    generated_ids = sequences[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _load_model_and_tokenizer(model_name_or_path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.to(device)
    return model, tokenizer


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Run CODI inference and latent-token sanity check.")
    parser.add_argument("model", help="Model name or path")
    parser.add_argument("prompt", help="Prompt text")
    parser.add_argument("--num-latents", type=int, default=6, help="Number of latent iterations")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--device", default=None, help="Device (e.g., cuda, cpu). Defaults to cuda if available.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model_and_tokenizer(args.model, device)

    result, token_strings, check = codi_inference_with_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_latent_iterations=args.num_latents,
        greedy=True,
    )

    prompt_len = tokenizer(args.prompt, return_tensors="pt")["input_ids"].shape[1]
    decoded = _decode_generated(tokenizer, result["sequences"], prompt_len)

    print("Generated output:")
    print(decoded)
    print("")
    print("Latent tokens:", " ".join(token_strings))
    print("Sanity check ok:", check.ok)
    print("Max abs diff:", check.max_abs_diff)


if __name__ == "__main__":
    _main()
