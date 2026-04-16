#!/usr/bin/env python3
"""
Paper 14 — Behavioral Geometry: Orthogonality of Persona and Reasoning Vectors

Tests whether SEAL-style reasoning vectors are orthogonal to persona vectors
(helpfulness, sycophancy, confidence, verbosity) in Gemma 4's activation space.

Key question: Can we compose trait steering and reasoning steering without 
interference? Near-orthogonal = safe to compose. Correlated = entangled risk.

Outputs:
  - results/vectors/reasoning_vectors.npy (if not exists)
  - results/behavioral_geometry.json — 5×5 cosine similarity matrix
  - results/figures/behavioral_geometry.png — heatmap visualization

Usage:
  python experiments/07_paper14.py --step dataset   # Generate reasoning prompts
  python experiments/07_paper14.py --step extract   # Extract reasoning vector
  python experiments/07_paper14.py --step analyze   # Compute geometry
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract import load_model, extract_from_jsonl, compute_persona_vector
from src.probe import probe_trait

VECTORS_DIR = ROOT / "results" / "vectors"
ACTIVATIONS_DIR = ROOT / "results" / "activations"
PROMPTS_DIR = ROOT / "data" / "prompts"

ALL_TRAITS = ["helpfulness", "sycophancy", "confidence", "verbosity", "reasoning"]
# Peak layers from Paper 10 + reasoning (to be determined)
PEAK_LAYERS = {
    "helpfulness": 0,
    "sycophancy": 1,
    "confidence": 3,
    "verbosity": 12,
    # reasoning peak will be detected
}


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def step_dataset():
    """Generate reasoning prompts (adds to existing dataset.py output)."""
    from src.dataset import main as dataset_main
    dataset_main()
    print(f"\nReasoning prompts added to {PROMPTS_DIR}/")
    print("Run: python experiments/07_paper14.py --step extract")


def step_extract():
    """Extract reasoning vector from Gemma 4-it."""
    print("Paper 14 — Extracting reasoning vector...")
    print("=" * 60)
    
    jsonl_path = PROMPTS_DIR / "reasoning.jsonl"
    if not jsonl_path.exists():
        print(f"Missing: {jsonl_path}")
        print("Run: python experiments/07_paper14.py --step dataset")
        sys.exit(1)
    
    model, tokenizer = load_model("mlx-community/gemma-4-e4b-it-4bit")
    
    pos_acts, neg_acts = extract_from_jsonl(
        model, tokenizer, jsonl_path, ACTIVATIONS_DIR, "reasoning"
    )
    reasoning_vec = compute_persona_vector(pos_acts, neg_acts)
    
    # Save
    np.save(VECTORS_DIR / "reasoning_vectors.npy", reasoning_vec)
    print(f"\nSaved → {VECTORS_DIR}/reasoning_vectors.npy")
    
    # Find peak layer
    accs = probe_trait(pos_acts, neg_acts, "reasoning")
    peak = int(np.argmax(accs))
    print(f"Reasoning peaks at layer {peak} (acc={accs[peak]:.3f})")
    print("\nRun: python experiments/07_paper14.py --step analyze")


def step_analyze():
    """Compute 5×5 behavioral geometry matrix."""
    print("Paper 14 — Behavioral Geometry Analysis")
    print("=" * 60)
    
    # Load all vectors
    vectors = {}
    peak_layers = {}
    
    for trait in ALL_TRAITS:
        vec_path = VECTORS_DIR / f"{trait}_vectors.npy"
        if not vec_path.exists():
            print(f"Missing: {vec_path}")
            if trait == "reasoning":
                print("Run: python experiments/07_paper14.py --step extract")
            sys.exit(1)
        
        vectors[trait] = np.load(vec_path)  # (n_layers, hidden_dim)
        
        # Determine peak layer
        pos_acts = np.load(ACTIVATIONS_DIR / f"{trait}_positive.npy")
        neg_acts = np.load(ACTIVATIONS_DIR / f"{trait}_negative.npy")
        accs = probe_trait(pos_acts, neg_acts, trait)
        peak_layers[trait] = int(np.argmax(accs))
    
    print("\nPeak layers detected:")
    for trait, layer in peak_layers.items():
        print(f"  {trait:15s}: layer {layer}")
    
    # Compute 5×5 cosine similarity matrix
    print("\nComputing cosine similarities...")
    n_traits = len(ALL_TRAITS)
    sim_matrix = np.zeros((n_traits, n_traits))
    
    for i, trait_i in enumerate(ALL_TRAITS):
        for j, trait_j in enumerate(ALL_TRAITS):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i < j:
                # Get vectors at respective peak layers
                vec_i = vectors[trait_i][peak_layers[trait_i]]
                vec_j = vectors[trait_j][peak_layers[trait_j]]
                sim = cos_sim(vec_i, vec_j)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
    
    # Flag entangled pairs (|cosine| > 0.3)
    print("\n" + "=" * 60)
    print("ENTANGLEMENT ANALYSIS (|cosine| > 0.3)")
    print("=" * 60)
    
    entangled = []
    orthogonal = []
    
    for i, trait_i in enumerate(ALL_TRAITS):
        for j, trait_j in enumerate(ALL_TRAITS):
            if i < j:
                sim = sim_matrix[i, j]
                pair = f"{trait_i} ↔ {trait_j}"
                if abs(sim) > 0.3:
                    entangled.append((pair, sim))
                else:
                    orthogonal.append((pair, sim))
    
    if entangled:
        print("\n⚠️  ENTANGLED pairs (steering may interfere):")
        for pair, sim in sorted(entangled, key=lambda x: abs(x[1]), reverse=True):
            print(f"   {pair:35s} cos={sim:+.3f}")
    
    if orthogonal:
        print("\n✓ ORTHOGONAL pairs (safe to compose):")
        for pair, sim in sorted(orthogonal, key=lambda x: abs(x[1])):
            print(f"   {pair:35s} cos={sim:+.3f}")
    
    # Specific hypothesis tests
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTS")
    print("=" * 60)
    
    tests = [
        ("reasoning", "confidence", "More reasoning = more calibrated (anti-hedging)"),
        ("reasoning", "verbosity", "Step-by-step = more words"),
        ("reasoning", "sycophancy", "Reasoning catches wrong claims"),
        ("reasoning", "helpfulness", "Being helpful ≠ reasoning more"),
    ]
    
    for t1, t2, desc in tests:
        i, j = ALL_TRAITS.index(t1), ALL_TRAITS.index(t2)
        sim = sim_matrix[i, j]
        status = "CORRELATED" if abs(sim) > 0.3 else "ORTHOGONAL"
        direction = "positive" if sim > 0 else "negative"
        print(f"\n{desc}")
        print(f"  {t1} ↔ {t2}: cos={sim:+.3f} → {status} ({direction})")
    
    # Save results
    results = {
        "traits": ALL_TRAITS,
        "peak_layers": peak_layers,
        "cosine_matrix": sim_matrix.tolist(),
        "entangled_pairs": [(p, float(s)) for p, s in entangled],
        "orthogonal_pairs": [(p, float(s)) for p, s in orthogonal],
        "interpretation": {
            "safe_to_compose": [p for p, _ in orthogonal],
            "risk_of_interference": [p for p, _ in entangled],
        }
    }
    
    out_json = ROOT / "results" / "behavioral_geometry.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_json}")
    
    # Plot heatmap
    plot_geometry_heatmap(sim_matrix, ALL_TRAITS, entangled)


def plot_geometry_heatmap(sim_matrix: np.ndarray, traits: list, entangled: list):
    """Create 5×5 behavioral geometry heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use RdBu_r colormap: blue=negative, white=0, red=positive
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    
    # Ticks
    ax.set_xticks(np.arange(len(traits)))
    ax.set_yticks(np.arange(len(traits)))
    ax.set_xticklabels(traits, rotation=45, ha="right")
    ax.set_yticklabels(traits)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("cosine similarity", rotation=-90, va="bottom", fontsize=12)
    
    # Annotate cells
    for i in range(len(traits)):
        for j in range(len(traits)):
            val = sim_matrix[i, j]
            # Highlight entangled pairs
            is_entangled = abs(val) > 0.3 and i != j
            color = "white" if abs(val) > 0.5 else "black"
            weight = "bold" if is_entangled else "normal"
            fontsize = 11 if is_entangled else 10
            
            ax.text(j, i, f"{val:.2f}", 
                   ha="center", va="center", 
                   color=color, fontsize=fontsize, weight=weight)
    
    # Title
    ax.set_title("Behavioral Geometry: Persona × Reasoning Vector Orthogonality\n(Gemma 4)", 
                 fontsize=14, pad=20)
    
    # Add annotation
    if entangled:
        entangled_text = "⚠️ Entangled (risk): " + ", ".join([p.split(" ↔ ")[0][:4] + "↔" + p.split(" ↔ ")[1][:4] 
                                                               for p, _ in entangled[:3]])
        fig.text(0.5, 0.02, entangled_text, ha="center", fontsize=10, 
                style="italic", color="darkred")
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    
    out_fig = ROOT / "results" / "figures" / "behavioral_geometry.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["dataset", "extract", "analyze"], required=True)
    args = parser.parse_args()
    
    if args.step == "dataset":
        step_dataset()
    elif args.step == "extract":
        step_extract()
    elif args.step == "analyze":
        step_analyze()


if __name__ == "__main__":
    main()
