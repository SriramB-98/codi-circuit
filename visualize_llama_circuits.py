#!/usr/bin/env python3
"""Visualize circuits for Llama 3.2 1B and CODI variants using circuit-tracer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from circuit_tracer import ReplacementModel, attribute  # noqa: E402
from circuit_tracer.utils import create_graph_files  # noqa: E402
from circuit_tracer.frontend.local_server import serve  # noqa: E402

from codi_inference import codi_inference_with_tokens, _load_prj

LLAMA_PLT_TRANSCODER_SET = "mntss/transcoder-Llama-3.2-1B"


SUPPORTED_MODELS = {
    "meta-llama/Llama-3.2-1B",
    "zen-E/CODI-llama3.2-1b-Instruct",
    "bcywinski/codi_llama1b-answer_only",
}

def _get_prj(model_name, model):
    if "bcywinski" in model_name:
        return _load_prj("./checkpoints/bcywinski/codi_llama1b-answer_only/prj.pt", None, model)
    else:
        raise ValueError(f"No known PRJ path for model {model_name}")

def is_codi_model(name: str) -> bool:
    lower = name.lower()
    return "codi" in lower

def default_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def slugify_model(model_name: str) -> str:
    safe = model_name.replace("/", "-").replace(" ", "-").lower()
    # ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{safe}"


def generate_continuation(
    prompt: str,
    model: str,
    tokenizer, 
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(input_ids=input_ids, **gen_kwargs)

    continuation_ids = output_ids[0, input_ids.shape[1] :]
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    del model, tokenizer
    return full_text, continuation


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize circuits with circuit-tracer.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B", help="HF model name")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument(
        "--transcoder-set",
        default=LLAMA_PLT_TRANSCODER_SET,
        help="HuggingFace transcoder repo (PLT/CLT) to use",
    )
    parser.add_argument("--backend", default="nnsight", choices=["nnsight", "transformerlens"])
    parser.add_argument("--prompt", required=True, help="Prompt to attribute")
    parser.add_argument("--slug", default=None, help="Slug for the graph (default: model + timestamp)")
    parser.add_argument("--graph-dir", default="graph_data/", help="Directory for graph JSON files")
    parser.add_argument("--graph-output-pt", default=None, help="Optional .pt output for raw graph")
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--max-n-logits", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--server", action="store_true", help="Start local visualization server")
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--num-latents", type=int, default=6, help="Number of CODI latents to use (if CODI model)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached data")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--codi-latent-path", default=None, help="Path to a torch file with CODI latent")

    args = parser.parse_args()

    if args.model not in SUPPORTED_MODELS:
        print("[warning] Model not in supported list; proceeding anyway.")

    dtype = default_dtype()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        # torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    slug = args.slug or slugify_model(args.model)
    graph_dir = Path(args.graph_dir) / slug
    graph_dir.mkdir(parents=True, exist_ok=True)

    if args.max_new_tokens == 0:
        attribution_prompt = args.prompt
    elif not is_codi_model(args.model):
        base_model.to(dtype)
        cache_path = f"cached_responses/{slug}.generation.json"
        if cache_path.exists() and not args.overwrite:
            print(f"Loading cached generation from {slug}.generation.json")
            with (cache_path).open("r", encoding="utf-8") as f:
                generation_data = json.load(f)
            full_text = generation_data["full_text"]
        else:
            full_text, continuation = generate_continuation(
                args.prompt,
                model=base_model,
                tokenizer=tokenizer,
                dtype=dtype,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": args.model,
                        "prompt": args.prompt,
                        "continuation": continuation,
                        "full_text": full_text,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    f,
                    indent=2,
                )
        attribution_prompt = full_text
    else:
        
        #!!! Mutates model in-place to add extra embeddings !!!
        result, token_strings, string_check, string_text = codi_inference_with_tokens(
            model=base_model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            num_latent_iterations=args.num_latents,
            greedy=True,
            sot_token_id=tokenizer.convert_tokens_to_ids("<|bocot|>"),
            eot_token_id=tokenizer.convert_tokens_to_ids("<|eocot|>"),
            projection=_get_prj(args.model, base_model),
        )
        del result
        print("Latent tokens:", " ".join(token_strings))
        print("String sanity check ok:", string_check.ok)
        print("String max abs diff:", string_check.max_abs_diff)
        print("String sanity check text:")
        print(string_text)
        attribution_prompt = string_text

        base_model.to(dtype)

    model = ReplacementModel.from_pretrained(
        args.model_name,
        args.transcoder_set,
        backend=args.backend,
        hf_model=base_model
        # dtype=dtype,
    )
    model.eval()
    breakpoint()

    graph = attribute(
        prompt=attribution_prompt,
        model=model,
        max_n_logits=args.max_n_logits,
        batch_size=args.batch_size,
        offload="disk"
    )

    if args.graph_output_pt:
        graph.to_pt(args.graph_output_pt)

    create_graph_files(
        graph,
        slug=slug,
        output_path=str(graph_dir)+'/',
        scan=args.transcoder_set,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )

    if args.server:
        del model
        server = serve(data_dir=str(graph_dir), port=args.port)
        print(f"Server running at http://localhost:{args.port}")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()


if __name__ == "__main__":
    main()
