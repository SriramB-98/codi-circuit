#!/usr/bin/env python3
"""
Load a CODI checkpoint, report missing/unexpected keys, and optionally
merge the PEFT adapter into the base model and save in HF format.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class CODI(nn.Module):
    """Minimal CODI wrapper to match checkpoint keys."""

    def __init__(
        self,
        codi: nn.Module,
        tokenizer: AutoTokenizer,
        num_latent: int = 6,
        use_prj: bool = True,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.codi = codi
        self.tokenizer = tokenizer
        self.num_latent = num_latent
        self.use_prj = use_prj
        self.pad_token_id = pad_token_id

        if use_prj:
            hidden_size = codi.config.hidden_size
            self.prj = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.prj.add_module("ln", nn.LayerNorm(hidden_size))
        else:
            self.prj = None


def _resolve_checkpoint_file(checkpoint_path: str, checkpoint_save_path: str) -> Path:
    os.makedirs(checkpoint_save_path, exist_ok=True)
    possible_files = [
        "model.safetensors",
        "adapter_model.safetensors",
        "pytorch_model.bin",
        "adapter_model.bin",
    ]

    for filename in possible_files:
        local_path = Path(checkpoint_save_path) / filename
        if local_path.exists():
            return local_path
        try:
            print(f"Trying to download {filename} from {checkpoint_path}...")
            hf_hub_download(
                repo_id=checkpoint_path,
                filename=filename,
                local_dir=checkpoint_save_path,
            )
            if local_path.exists():
                print(f"Successfully downloaded {filename}")
                return local_path
        except Exception:
            print(f"  {filename} not found, trying next...")
            continue

    available = list_repo_files(checkpoint_path)
    raise FileNotFoundError(
        f"Could not find checkpoint in {checkpoint_path}. "
        f"Available files: {available}"
    )


def _load_state_dict(checkpoint_file: Path) -> dict:
    print(f"Loading weights from {checkpoint_file}...")
    if str(checkpoint_file).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(str(checkpoint_file))
    return torch.load(str(checkpoint_file), map_location="cpu")


def _build_codi_model(
    model_name_or_path: str,
    lora_r: int,
    lora_alpha: int,
    num_latent: int,
    use_prj: bool,
    device: str,
    dtype: str,
):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|bocot|>", "<|eocot|>"]
    })
    base_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    codi_model = CODI(
        codi=model,
        tokenizer=tokenizer,
        num_latent=num_latent,
        use_prj=use_prj,
        pad_token_id=tokenizer.pad_token_id,
    )

    if codi_model.use_prj and codi_model.prj is not None:
        codi_model.prj.to(dtype=torch_dtype)

    return codi_model, tokenizer


def _print_key_report(missing, unexpected):
    if missing:
        print(f"Missing keys: {len(missing)}")
        for key in missing:
            print(f"  {key}")
    else:
        print("Missing keys: 0")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for key in unexpected:
            print(f"  {key}")
    else:
        print("Unexpected keys: 0")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load CODI checkpoint, report missing/unexpected keys, and optionally consolidate."
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=False, default=None)
    parser.add_argument("--checkpoint_save_path", default=None)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--num_latent", type=int, default=6)
    parser.add_argument("--use_prj", action="store_true")
    parser.add_argument("--no_use_prj", dest="use_prj", action="store_false")
    parser.set_defaults(use_prj=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--strict", action="store_true")

    args = parser.parse_args()

    checkpoint_save_path = args.checkpoint_save_path
    if checkpoint_save_path is None:
        checkpoint_save_path = f"./checkpoints/{args.checkpoint_path.replace('/', '_')}"

    checkpoint_file = _resolve_checkpoint_file(args.checkpoint_path, checkpoint_save_path)
    state_dict = _load_state_dict(checkpoint_file)

    codi_model, tokenizer = _build_codi_model(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_latent=args.num_latent,
        use_prj=args.use_prj,
        device=args.device,
        dtype=args.dtype,
    )

    missing, unexpected = codi_model.load_state_dict(state_dict, strict=args.strict)
    _print_key_report(missing, unexpected)

    breakpoint()
    
    if args.output_dir is None:
        print("No --output_dir provided; skipping consolidation.")
        return 0

    answer = input("Proceed with consolidation (merge PEFT + save HF model)? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Skipping consolidation.")
        return 0

    output_dir = Path(args.output_dir) / f"merged_{args.checkpoint_path.replace('/', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Merging PEFT adapter into base model...")
    merged = codi_model.codi.merge_and_unload()

    print(f"Saving merged model + tokenizer to {output_dir}...")
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done. You can now load with AutoModelForCausalLM.from_pretrained and AutoTokenizer.from_pretrained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
