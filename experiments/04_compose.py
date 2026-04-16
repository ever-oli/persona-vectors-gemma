#!/usr/bin/env python3
"""
Phase 4 — Composability Experiment (Novel Contribution)

Tests whether persona vectors compose additively:
  steer(helpful + confident) ≈ steer(helpful) + steer(confident)

Three conditions:
  A: helpful alone
  B: confident alone
  C: helpful + confident simultaneously

Measures via proxy metrics and computes a composability score.

Usage:
    python experiments/04_compose.py
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract import load_model
from src.steer import generate_steered, generate_composed
import numpy as np
import matplotlib.pyplot as plt

VECTORS_DIR = ROOT / "results" / "vectors"
RESULTS     = ROOT / "results"
DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"

# Prompts for composability test (neutral prompts where both traits are observable)
COMPOSE_PROMPTS = [
    "<start_of_turn>user\nShould I use microservices or a monolith for my startup?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat's the most important thing to learn in machine learning?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nIs remote work better than in-office for software engineers?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat database should I use for a new web app?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nHow do I get better at coding?\n<end_of_turn>\n<start_of_turn>model\n",
]


# ---------------------------------------------------------------------------
# Proxy metrics
# ---------------------------------------------------------------------------

HEDGE_WORDS = {"maybe", "perhaps", "possibly", "might", "could", "probably",
               "i think", "i believe", "not sure", "uncertain", "it depends",
               "generally", "often", "sometimes", "typically", "usually"}


def helpfulness_score(text: str) -> float:
    """Word count proxy: more words = more helpful."""
    return len(text.split())


def confidence_score(text: str) -> float:
    """Negative hedge word rate: fewer hedges = more confident."""
    words = text.lower()
    count = sum(1 for hw in HEDGE_WORDS if hw in words)
    return -count  # more negative = less confident, so higher = more confident


def sycophancy_score(text: str) -> float:
    """Agreement markers rate (positive = more sycophantic)."""
    agree = ["you're right", "absolutely", "great point", "exactly", "indeed", "of course"]
    return sum(1 for a in agree if a in text.lower())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    peak_path = RESULTS / "peak_layers.json"
    if not peak_path.exists():
        print("Run 02_probe.py first."); sys.exit(1)

    peak_layers = json.loads(peak_path.read_text())
    model, tokenizer = load_model(DEFAULT_MODEL)

    traits_to_compose = ["helpfulness", "confidence"]  # primary test pair
    alpha = 12.0

    # Load vectors
    vectors = {}
    for trait in traits_to_compose:
        vec_path = VECTORS_DIR / f"{trait}_vectors.npy"
        if not vec_path.exists():
            print(f"Missing vector: {vec_path}"); sys.exit(1)
        vectors[trait] = np.load(vec_path)

    # Use helpfulness peak layer as the shared target
    target_layer = peak_layers.get("helpfulness", 14)

    results = {
        "helpfulness_only": [],
        "confidence_only":  [],
        "composed":         [],
        "baseline":         [],
    }

    metric_fns = {
        "helpfulness": helpfulness_score,
        "confidence":  confidence_score,
    }

    print(f"Composability test: {' + '.join(traits_to_compose)} at layer {target_layer}")
    print(f"Prompts: {len(COMPOSE_PROMPTS)}\n")

    for i, prompt in enumerate(COMPOSE_PROMPTS):
        print(f"Prompt {i+1}/{len(COMPOSE_PROMPTS)}: {prompt.strip()[:60]}...")

        baseline = generate_steered(model, tokenizer, prompt,
                                    vectors["helpfulness"], target_layer,
                                    alpha=0.0, max_tokens=120)

        helpful_only = generate_steered(model, tokenizer, prompt,
                                        vectors["helpfulness"], target_layer,
                                        alpha=alpha, max_tokens=120)

        confident_only = generate_steered(model, tokenizer, prompt,
                                          vectors["confidence"], target_layer,
                                          alpha=alpha, max_tokens=120)

        composed = generate_composed(model, tokenizer, prompt,
                                     {t: vectors[t] for t in traits_to_compose},
                                     {"helpfulness": alpha, "confidence": alpha},
                                     target_layer=target_layer, max_tokens=120)

        results["baseline"].append({
            "text": baseline,
            "helpfulness": helpfulness_score(baseline),
            "confidence":  confidence_score(baseline),
        })
        results["helpfulness_only"].append({
            "text": helpful_only,
            "helpfulness": helpfulness_score(helpful_only),
            "confidence":  confidence_score(helpful_only),
        })
        results["confidence_only"].append({
            "text": confident_only,
            "helpfulness": helpfulness_score(confident_only),
            "confidence":  confidence_score(confident_only),
        })
        results["composed"].append({
            "text": composed,
            "helpfulness": helpfulness_score(composed),
            "confidence":  confidence_score(composed),
        })

    # Compute composability scores
    def avg_metric(cond, metric):
        return np.mean([r[metric] for r in results[cond]])

    summary = {}
    for metric in ["helpfulness", "confidence"]:
        A = avg_metric("helpfulness_only", metric)
        B = avg_metric("confidence_only",  metric)
        C = avg_metric("composed",         metric)
        base = avg_metric("baseline", metric)

        # If vectors compose: C ≈ A + B - base (additive, baseline-corrected)
        predicted_additive = A + B - base
        compose_score = 1 - abs(C - predicted_additive) / (abs(predicted_additive - base) + 1e-6)

        summary[metric] = {
            "baseline": round(base, 3),
            "A_only":   round(A, 3),
            "B_only":   round(B, 3),
            "composed": round(C, 3),
            "predicted_additive": round(predicted_additive, 3),
            "composability_score": round(float(compose_score), 3),
        }
        print(f"\n{metric}:")
        print(f"  baseline:           {base:.2f}")
        print(f"  helpful only:       {A:.2f}")
        print(f"  confident only:     {B:.2f}")
        print(f"  composed:           {C:.2f}")
        print(f"  predicted additive: {predicted_additive:.2f}")
        print(f"  composability:      {compose_score:.3f}  (1.0 = perfect additive)")

    # Save results
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "composability.json").write_text(json.dumps({
        "summary": summary,
        "target_layer": target_layer,
        "alpha": alpha,
        "n_prompts": len(COMPOSE_PROMPTS),
    }, indent=2))
    print(f"\nSaved → {RESULTS}/composability.json")

    # Plot
    _plot_composability(summary, RESULTS / "figures" / "composability.png")
    print("All done!")


def _plot_composability(summary: dict, out_path: Path) -> None:
    metrics = list(summary.keys())
    conditions = ["baseline", "A_only", "B_only", "composed", "predicted_additive"]
    labels = ["baseline", "helpful only", "confident only", "composed (actual)", "predicted\n(additive)"]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#F44336", "#FF9800"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = [summary[metric][c] for c in conditions]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{metric} proxy\ncomposability={summary[metric]['composability_score']:.2f}", fontsize=11)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9)

    fig.suptitle("Persona Vector Composability — Gemma 4", fontsize=13, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
