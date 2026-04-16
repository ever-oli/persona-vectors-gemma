#!/usr/bin/env python3
"""
Paper 13 — Alignment Depth Taxonomy

Three-way comparison to test if Paper 12's layer-0 finding was a template artifact:
  1. IT model (chat template) — results/vectors/
  2. Base model (chat template) — results/vectors_base/
  3. Base model (plain text) — results/vectors_base_plain/ [NEW]

Outputs:
  - results/alignment_taxonomy.json — cosine similarities, peak layer shifts
  - results/figures/alignment_taxonomy.png — 4×3 heatmap

Usage:
  python experiments/06_paper13.py --step dataset    # Generate plain-text prompts
  python experiments/06_paper13.py --step extract    # Extract from base model (~45 min)
  python experiments/06_paper13.py --step compare    # Three-way comparison
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract import load_model, extract_from_jsonl, compute_persona_vector

VECTORS_IT = ROOT / "results" / "vectors"
VECTORS_BASE = ROOT / "results" / "vectors_base"
VECTORS_BASE_PLAIN = ROOT / "results" / "vectors_base_plain"
ACTIVATIONS_BASE_PLAIN = ROOT / "results" / "activations_base_plain"
PROMPTS_BASE = ROOT / "data" / "prompts_base"
TRAITS = ["helpfulness", "sycophancy", "confidence", "verbosity"]


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_vectors(path: Path, trait: str) -> np.ndarray:
    """Load persona vector for a trait."""
    return np.load(path / f"{trait}_vectors.npy")  # (n_layers, hidden_dim)


def probe_accuracy(acts: np.ndarray, labels: np.ndarray) -> list[float]:
    """Train probe per layer, return accuracies."""
    n_layers = acts.shape[1]
    accs = []
    for layer in range(n_layers):
        X = acts[:, layer, :]  # (n_samples, hidden_dim)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, labels)
        preds = clf.predict(X)
        accs.append(accuracy_score(labels, preds))
    return accs


def find_peak_layer(acts: np.ndarray) -> tuple[int, float]:
    """Find layer with highest probe accuracy."""
    n = acts.shape[0] // 2
    labels = np.array([1] * n + [0] * n)
    accs = probe_accuracy(acts, labels)
    peak = int(np.argmax(accs))
    return peak, accs[peak]


def step_dataset():
    """Generate plain-text dataset for base model."""
    print("Generating plain-text dataset...")
    from src.dataset_base import main
    main()
    print(f"\nDone. Prompts saved to {PROMPTS_BASE}/")


def step_extract():
    """Extract vectors from base model with plain-text prompts."""
    print("Paper 13 — Extracting base model with plain-text prompts...")
    print(f"Model: mlx-community/gemma-4-e4b-4bit")
    print()

    # Generate dataset if needed
    for trait in TRAITS:
        jsonl_path = PROMPTS_BASE / f"{trait}.jsonl"
        if not jsonl_path.exists():
            print(f"Plain-text prompts missing: {jsonl_path}")
            print("Run: python experiments/06_paper13.py --step dataset")
            sys.exit(1)

    model, tokenizer = load_model("mlx-community/gemma-4-e4b-4bit")

    persona_vectors = {}

    for trait in TRAITS:
        jsonl_path = PROMPTS_BASE / f"{trait}.jsonl"
        pos_acts, neg_acts = extract_from_jsonl(
            model, tokenizer, jsonl_path, ACTIVATIONS_BASE_PLAIN, trait
        )
        persona_vectors[trait] = compute_persona_vector(pos_acts, neg_acts)

    # Save vectors
    VECTORS_BASE_PLAIN.mkdir(parents=True, exist_ok=True)
    for trait, vec in persona_vectors.items():
        path = VECTORS_BASE_PLAIN / f"{trait}_vectors.npy"
        np.save(path, vec)
        print(f"Saved → {path}  shape={vec.shape}")

    print("\nPlain-text extraction done. Run --step compare next.")


def step_compare():
    """Three-way comparison and taxonomy analysis."""
    print("Paper 13 — Three-way comparison (IT vs template-base vs plain-base)")
    print()

    # Check all vectors exist
    for trait in TRAITS:
        for path, name in [
            (VECTORS_IT, "IT"),
            (VECTORS_BASE, "template-base"),
            (VECTORS_BASE_PLAIN, "plain-base"),
        ]:
            vec_path = path / f"{trait}_vectors.npy"
            if not vec_path.exists():
                print(f"Missing: {vec_path} ({name})")
                if name == "plain-base":
                    print("Run: python experiments/06_paper13.py --step extract")
                sys.exit(1)

    results = {}

    # Load activations for peak layer analysis
    acts_it = {}
    acts_base = {}
    acts_base_plain = {}

    for trait in TRAITS:
        acts_it[trait] = np.load(ROOT / "results" / "activations" / f"{trait}_positive.npy")
        acts_base[trait] = np.load(ROOT / "results" / "activations_base" / f"{trait}_positive.npy")
        acts_base_plain[trait] = np.load(ACTIVATIONS_BASE_PLAIN / f"{trait}_positive.npy")

    for trait in TRAITS:
        print(f"\n=== {trait.upper()} ===")

        # Load vectors
        vec_it = load_vectors(VECTORS_IT, trait)
        vec_base = load_vectors(VECTORS_BASE, trait)
        vec_base_plain = load_vectors(VECTORS_BASE_PLAIN, trait)

        # Find peak layers
        peak_it, _ = find_peak_layer(acts_it[trait])
        peak_base, _ = find_peak_layer(acts_base[trait])
        peak_base_plain, _ = find_peak_layer(acts_base_plain[trait])

        # Cosine similarities
        cos_template_to_it = cos_sim(vec_base[peak_base], vec_it[peak_it])
        cos_plain_to_it = cos_sim(vec_base_plain[peak_base_plain], vec_it[peak_it])
        cos_plain_to_template = cos_sim(vec_base_plain[peak_base_plain], vec_base[peak_base])

        print(f"  Peak layers: base-plain={peak_base_plain}, base-template={peak_base}, IT={peak_it}")
        print(f"  Cosine template-base → IT:  {cos_template_to_it:+.4f}")
        print(f"  Cosine plain-base → IT:     {cos_plain_to_it:+.4f}")
        print(f"  Cosine plain-base → template: {cos_plain_to_template:+.4f}")

        # Interpretation
        if cos_plain_to_it < 0.3:
            depth = "DEEP restructuring"
        elif cos_plain_to_it < 0.7:
            depth = "MODERATE restructuring"
        else:
            depth = "SHALLOW (mostly unchanged)"

        print(f"  → {depth}")

        results[trait] = {
            "peak_layer": {
                "it": int(peak_it),
                "base_template": int(peak_base),
                "base_plain": int(peak_base_plain),
            },
            "cosine_similarity": {
                "template_base_to_it": cos_template_to_it,
                "plain_base_to_it": cos_plain_to_it,
                "plain_to_template_base": cos_plain_to_template,
            },
            "alignment_depth_category": depth,
        }

    # Save results
    out_json = ROOT / "results" / "alignment_taxonomy.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_json}")

    # Generate heatmap
    plot_taxonomy_heatmap(results)

    # Summary
    print("\n" + "=" * 60)
    print("ALIGNMENT DEPTH TAXONOMY")
    print("=" * 60)
    for trait, data in results.items():
        cos = data["cosine_similarity"]["plain_base_to_it"]
        depth = data["alignment_depth_category"]
        peak_shift = data["peak_layer"]["it"] - data["peak_layer"]["base_plain"]
        print(f"  {trait:15s}  cos={cos:+.4f}  peak_shift={peak_shift:+2d}  → {depth}")

    # Artifact check
    if all(r["cosine_similarity"]["plain_base_to_it"] < 0.3 for r in results.values()):
        print("\n  → Paper 12 finding HOLDS: all traits restructure deeply")
    elif all(r["cosine_similarity"]["plain_base_to_it"] > 0.7 for r in results.values()):
        print("\n  → Paper 12 finding was ARTIFACT: template confused base model")
    else:
        print("\n  → MIXED: some traits restructure, others don't (most interesting)")


def plot_taxonomy_heatmap(results: dict):
    """Create 4×3 heatmap of cosine similarities."""
    traits = list(results.keys())
    cos_keys = ["template_base_to_it", "plain_base_to_it", "plain_to_template_base"]
    labels = ["template→IT", "plain→IT", "plain→template"]

    data = np.zeros((len(traits), len(cos_keys)))
    for i, trait in enumerate(traits):
        for j, key in enumerate(cos_keys):
            data[i, j] = results[trait]["cosine_similarity"][key]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(traits)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(traits)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("cosine similarity", rotation=-90, va="bottom")

    # Annotate cells
    for i in range(len(traits)):
        for j in range(len(labels)):
            val = data[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=12)

    ax.set_title("Alignment Depth Taxonomy: Base vs IT Cosine Similarity", fontsize=14)
    fig.tight_layout()

    out_fig = ROOT / "results" / "figures" / "alignment_taxonomy.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    print(f"Saved figure → {out_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["dataset", "extract", "compare"], required=True)
    args = parser.parse_args()

    if args.step == "dataset":
        step_dataset()
    elif args.step == "extract":
        step_extract()
    elif args.step == "compare":
        step_compare()


if __name__ == "__main__":
    main()
