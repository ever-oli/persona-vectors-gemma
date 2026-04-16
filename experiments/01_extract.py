#!/usr/bin/env python3
"""
Phase 1 — Activation Extraction

Run this first. Downloads Gemma 4, extracts hidden states for all traits.
Takes ~45 min on M4 Mac Mini for 10 pairs × 4 traits.

Usage:
    python experiments/01_extract.py
    python experiments/01_extract.py --model mlx-community/gemma-4-e4b-it-4bit --traits helpfulness verbosity
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract import load_model, extract_from_jsonl, compute_persona_vector, save_persona_vectors

ACTS_DIR    = ROOT / "results" / "activations"
VECTORS_DIR = ROOT / "results" / "vectors"
PROMPTS_DIR = ROOT / "data" / "prompts"

TRAITS = ["helpfulness", "sycophancy", "confidence", "verbosity"]
DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
FALLBACK_MODEL = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--traits", nargs="+", default=TRAITS)
    parser.add_argument("--fallback", action="store_true",
                        help="Use unsloth fallback model instead")
    args = parser.parse_args()

    model_id = FALLBACK_MODEL if args.fallback else args.model
    model, tokenizer = load_model(model_id)

    # Generate dataset if not already present
    for trait in args.traits:
        jsonl_path = PROMPTS_DIR / f"{trait}.jsonl"
        if not jsonl_path.exists():
            print(f"Prompt file missing: {jsonl_path}")
            print("Run: python src/dataset.py")
            sys.exit(1)

    persona_vectors = {}

    for trait in args.traits:
        jsonl_path = PROMPTS_DIR / f"{trait}.jsonl"
        pos_acts, neg_acts = extract_from_jsonl(model, tokenizer, jsonl_path, ACTS_DIR, trait)
        persona_vectors[trait] = compute_persona_vector(pos_acts, neg_acts)

    save_persona_vectors(persona_vectors, VECTORS_DIR)
    print("\nDone. Run experiments/02_probe.py next.")


if __name__ == "__main__":
    main()
