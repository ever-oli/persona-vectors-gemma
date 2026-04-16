#!/usr/bin/env python3
"""
Paper 12 — Alignment Effect on Persona Vectors: Gemma 4 base vs Gemma 4-it

Both models are Gemma 4 E4B, same architecture, same weights before RLHF.
The only difference is instruction tuning (alignment).

Questions:
  1. Did alignment MOVE the persona vector? (cosine similarity base vs it)
     - cos ≈ 1.0 → RLHF didn't change the direction → alignment is surface-level
     - cos ≈ 0.0 → RLHF restructured the representation → deep alignment
  2. Did probe accuracy change? (is the trait MORE linearly decodable after alignment?)
  3. Did the peak layer shift? (does RLHF push traits earlier or later in the network?)

This directly engages Paper 8 (Superficial Knowledge in Alignment, Chen et al. NAACL 2025):
  - Paper 8 found aligned outputs can be "superficial" (output changes, repr doesn't)
  - Paper 12 tests: do persona vectors move, or just the outputs?

Models:
  - Instruction-tuned (Paper 10, already done): mlx-community/gemma-4-e4b-it-4bit
  - Base (Paper 12, new):                       mlx-community/gemma-4-e4b-4bit

Usage:
    python experiments/05_cross_model.py --step extract   # download base, extract vectors
    python experiments/05_cross_model.py --step compare   # compare base vs it
    python experiments/05_cross_model.py                  # both
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

from src.extract import load_model, extract_from_jsonl, compute_persona_vector, save_persona_vectors
from src.probe import probe_trait, plot_probe_accuracy

TRAITS      = ["helpfulness", "sycophancy", "confidence", "verbosity"]
PROMPTS_DIR = ROOT / "data" / "prompts"          # reuse Gemma-it prompts (same format)

IT_ACTS     = ROOT / "results" / "activations"   # already computed in Paper 10
IT_VECS     = ROOT / "results" / "vectors"
IT_PEAKS    = ROOT / "results" / "peak_layers.json"
IT_PROBES   = ROOT / "results" / "probe_accuracy.json"

BASE_ACTS   = ROOT / "results" / "activations_base"
BASE_VECS   = ROOT / "results" / "vectors_base"
RESULTS     = ROOT / "results"

BASE_MODEL_ID = "mlx-community/gemma-4-e4b-4bit"
IT_MODEL_ID   = "mlx-community/gemma-4-e4b-it-4bit"


# ---------------------------------------------------------------------------
# Step 1: Extract base model activations
# ---------------------------------------------------------------------------

def step_extract():
    """Download Gemma 4 base and extract persona vectors."""
    print(f"Loading base model: {BASE_MODEL_ID}")
    print("Note: same prompts as Paper 10 — base model sees instruction-format text")
    print("      but was not trained to follow instructions. That's intentional.")
    print("      We want the representation of the same text, not instruction-following.\n")

    model, tokenizer = load_model(BASE_MODEL_ID)
    persona_vectors = {}

    for trait in TRAITS:
        jsonl_path = PROMPTS_DIR / f"{trait}.jsonl"
        if not jsonl_path.exists():
            print(f"Missing: {jsonl_path} — run src/dataset.py first")
            sys.exit(1)

        pos_acts, neg_acts = extract_from_jsonl(model, tokenizer, jsonl_path, BASE_ACTS, trait)
        persona_vectors[trait] = compute_persona_vector(pos_acts, neg_acts)

    save_persona_vectors(persona_vectors, BASE_VECS)
    print("\nBase model extraction done.")


# ---------------------------------------------------------------------------
# Step 2: Compare base vs it
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def step_compare():
    if not (IT_PEAKS).exists():
        print("Run 02_probe.py first."); sys.exit(1)

    it_peaks   = json.loads(IT_PEAKS.read_text())
    it_probes  = json.loads(IT_PROBES.read_text())

    # Compute base model probe accuracies
    print("Computing base model probe accuracies ...")
    base_probes = {}
    base_peaks  = {}

    for trait in TRAITS:
        pos_path = BASE_ACTS / f"{trait}_positive.npy"
        if not pos_path.exists():
            print(f"  Skipping {trait} — run --step extract first")
            continue

        pos_acts = np.load(pos_path)
        neg_acts = np.load(BASE_ACTS / f"{trait}_negative.npy")
        accs = probe_trait(pos_acts, neg_acts, f"{trait}_base")
        base_probes[trait] = accs.tolist()
        base_peaks[trait]  = int(np.argmax(accs))

    if not base_probes:
        print("No base activations found. Run --step extract first.")
        sys.exit(1)

    n_layers = len(list(base_probes.values())[0])

    # --- Cosine similarity: base vs it vector at each model's peak layer ---
    print("\n--- Vector Cosine Similarity (base vs it, at respective peak layers) ---")
    cos_results = {}

    for trait in TRAITS:
        if trait not in base_peaks:
            continue

        base_vec = np.load(BASE_VECS / f"{trait}_vectors.npy")[base_peaks[trait]]
        it_vec   = np.load(IT_VECS   / f"{trait}_vectors.npy")[it_peaks[trait]]

        # Same dimension (both Gemma 4, 2560) — direct cosine
        cos = cosine(base_vec, it_vec)
        cos_results[trait] = round(cos, 4)

        interpretation = (
            "SURFACE-LEVEL alignment (repr unchanged)" if cos > 0.9
            else "PARTIAL shift" if cos > 0.5
            else "DEEP alignment (repr restructured)"
        )
        print(f"  {trait:<15} cos={cos:+.4f}  → {interpretation}")

    # --- Peak layer shift ---
    print("\n--- Peak Layer Shift (base → it) ---")
    print(f"{'Trait':<15} {'Base peak':>10} {'It peak':>10} {'Shift':>8} {'Direction':>15}")
    print("-" * 62)

    layer_shift = {}
    for trait in TRAITS:
        if trait not in base_peaks:
            continue
        b = base_peaks[trait]
        i = it_peaks[trait]
        shift = i - b
        direction = "earlier" if shift < 0 else "later" if shift > 0 else "same"
        print(f"  {trait:<15} {b:>10} {i:>10} {shift:>+8} {direction:>15}")
        layer_shift[trait] = {"base": b, "it": i, "shift": shift}

    # --- Probe accuracy delta ---
    print("\n--- Probe Accuracy Delta (it - base, at peak layer) ---")
    probe_delta = {}
    for trait in TRAITS:
        if trait not in base_peaks:
            continue
        base_acc = max(base_probes[trait])
        it_acc   = max(it_probes[trait])
        delta    = it_acc - base_acc
        print(f"  {trait:<15} base={base_acc:.3f}  it={it_acc:.3f}  delta={delta:+.3f}")
        probe_delta[trait] = {"base_acc": round(base_acc, 4), "it_acc": round(it_acc, 4), "delta": round(delta, 4)}

    # --- Save ---
    results = {
        "cosine_similarity":  cos_results,
        "peak_layer_shift":   layer_shift,
        "probe_accuracy_delta": probe_delta,
        "base_peak_layers":   base_peaks,
        "it_peak_layers":     it_peaks,
        "interpretation": _interpret(cos_results, layer_shift, probe_delta),
    }
    out = RESULTS / "alignment_effect.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    # --- Figure ---
    _plot_comparison(
        base_probes, it_probes, base_peaks, it_peaks,
        cos_results,
        out_path=RESULTS / "figures" / "base_vs_it_probes.png",
    )

    print("\n--- Overall interpretation ---")
    print(results["interpretation"])


def _interpret(cos_results, layer_shift, probe_delta) -> str:
    avg_cos   = np.mean(list(cos_results.values())) if cos_results else 0
    avg_shift = np.mean([abs(v["shift"]) for v in layer_shift.values()]) if layer_shift else 0
    avg_delta = np.mean([v["delta"] for v in probe_delta.values()]) if probe_delta else 0

    lines = []
    if avg_cos > 0.85:
        lines.append(
            f"High cosine similarity ({avg_cos:.2f}) — RLHF did NOT significantly move persona vectors. "
            "Alignment appears surface-level: the representation is the same, only outputs changed. "
            "Consistent with Chen et al. 2025 (Paper 8) superficial alignment finding."
        )
    elif avg_cos > 0.5:
        lines.append(
            f"Moderate cosine similarity ({avg_cos:.2f}) — RLHF partially restructured representations. "
            "Some traits changed more than others (see per-trait breakdown)."
        )
    else:
        lines.append(
            f"Low cosine similarity ({avg_cos:.2f}) — RLHF significantly changed persona vector directions. "
            "Deep alignment: the model's internal trait representation was restructured."
        )

    if avg_delta > 0.05:
        lines.append(
            f"Probe accuracy increased after alignment (+{avg_delta:.3f} avg) — "
            "traits are MORE linearly decodable in the aligned model. "
            "Alignment sharpened trait representations."
        )
    elif avg_delta < -0.05:
        lines.append(
            f"Probe accuracy decreased after alignment ({avg_delta:.3f} avg) — "
            "traits are LESS linearly decodable. Alignment dispersed trait information."
        )

    if avg_shift > 2:
        lines.append(
            f"Peak layer shifted by {avg_shift:.1f} layers on average — "
            "alignment changed WHERE in the network traits are encoded."
        )

    return " ".join(lines)


def _plot_comparison(base_probes, it_probes, base_peaks, it_peaks, cos_results, out_path):
    traits = [t for t in TRAITS if t in base_probes and t in it_probes]
    if not traits:
        return

    fig, axes = plt.subplots(1, len(traits), figsize=(5 * len(traits), 4))
    if len(traits) == 1:
        axes = [axes]

    for ax, trait in zip(axes, traits):
        b_accs = np.array(base_probes[trait])
        i_accs = np.array(it_probes[trait])
        layers = list(range(len(b_accs)))

        ax.plot(layers, b_accs, color="#9E9E9E", label="base",  linewidth=2, linestyle="--")
        ax.plot(layers, i_accs, color="#4285F4", label="it",    linewidth=2)

        ax.axvline(base_peaks[trait], color="#9E9E9E", alpha=0.4, linestyle=":")
        ax.axvline(it_peaks[trait],   color="#4285F4", alpha=0.4, linestyle=":")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        cos = cos_results.get(trait, 0)
        ax.set_title(f"{trait}\ncos(base,it)={cos:+.3f}", fontsize=10)
        ax.set_xlabel("Layer", fontsize=9)
        if trait == traits[0]:
            ax.set_ylabel("Probe accuracy", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle("Persona Vectors: Base vs Instruction-Tuned Gemma 4\n(Does RLHF move the direction?)", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["extract", "compare", "both"], default="both")
    args = parser.parse_args()

    if args.step in ("extract", "both"):
        step_extract()
    if args.step in ("compare", "both"):
        step_compare()


if __name__ == "__main__":
    main()
