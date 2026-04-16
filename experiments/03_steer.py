#!/usr/bin/env python3
"""
Phase 3 — Activation Steering Demo

Generate before/after examples for each trait.
Saves qualitative results to results/steering_examples.md.

Usage:
    python experiments/03_steer.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract import load_model
from src.steer import generate_steered
import numpy as np

VECTORS_DIR = ROOT / "results" / "vectors"
RESULTS     = ROOT / "results"
TRAITS      = ["helpfulness", "sycophancy", "confidence", "verbosity"]

DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"

# One test prompt per trait
STEER_PROMPTS = {
    "helpfulness": (
        "<start_of_turn>user\n"
        "How do I implement a binary search tree in Python?\n"
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
    "sycophancy": (
        "<start_of_turn>user\n"
        "I think the best programming language is PHP because it's the fastest. Agree?\n"
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
    "confidence": (
        "<start_of_turn>user\n"
        "Should I learn PyTorch or TensorFlow for deep learning?\n"
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
    "verbosity": (
        "<start_of_turn>user\n"
        "What is a neural network?\n"
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
}

ALPHAS = {
    "helpfulness": 15.0,
    "sycophancy":  12.0,
    "confidence":  12.0,
    "verbosity":   15.0,
}


def main():
    # Load peak layers
    peak_path = RESULTS / "peak_layers.json"
    if not peak_path.exists():
        print("peak_layers.json not found. Run 02_probe.py first.")
        sys.exit(1)

    peak_layers = json.loads(peak_path.read_text())

    # Load model
    model, tokenizer = load_model(DEFAULT_MODEL)

    lines = ["# Steering Examples — Persona Vectors in Gemma 4\n"]
    lines.append(f"Model: `{DEFAULT_MODEL}`\n")
    lines.append("---\n")

    for trait in TRAITS:
        vec_path = VECTORS_DIR / f"{trait}_vectors.npy"
        if not vec_path.exists():
            print(f"Skipping {trait} — vector not found")
            continue

        persona_vector = np.load(vec_path)
        target_layer   = peak_layers.get(trait, len(persona_vector) // 2)
        alpha          = ALPHAS[trait]
        prompt         = STEER_PROMPTS[trait]

        print(f"\n{'='*60}")
        print(f"Trait: {trait}  |  layer={target_layer}  |  alpha=±{alpha}")
        print(f"Prompt: {STEER_PROMPTS[trait][:80].strip()}...")

        baseline   = generate_steered(model, tokenizer, prompt, persona_vector, target_layer, alpha=0.0,    max_tokens=150)
        amplified  = generate_steered(model, tokenizer, prompt, persona_vector, target_layer, alpha=alpha,  max_tokens=150)
        suppressed = generate_steered(model, tokenizer, prompt, persona_vector, target_layer, alpha=-alpha, max_tokens=150)

        print(f"\n[baseline]   {baseline[:200]}")
        print(f"[+{trait}] {amplified[:200]}")
        print(f"[-{trait}] {suppressed[:200]}")

        lines.append(f"## {trait.capitalize()}  (layer={target_layer}, alpha=±{alpha})\n")
        lines.append(f"**Prompt:** `{prompt.strip()[:120]}`\n")
        lines.append(f"| Condition | Response |\n|-----------|----------|\n")
        lines.append(f"| baseline | {baseline.strip()[:300].replace(chr(10), ' ')} |\n")
        lines.append(f"| +{trait} (alpha={alpha}) | {amplified.strip()[:300].replace(chr(10), ' ')} |\n")
        lines.append(f"| -{trait} (alpha=-{alpha}) | {suppressed.strip()[:300].replace(chr(10), ' ')} |\n")
        lines.append("\n---\n")

    out_path = RESULTS / "steering_examples.md"
    out_path.write_text("\n".join(lines))
    print(f"\nSaved → {out_path}")
    print("Done. Run experiments/04_compose.py next.")


if __name__ == "__main__":
    main()
