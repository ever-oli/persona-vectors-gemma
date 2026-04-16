#!/usr/bin/env python3
"""
Phase 2 — Linear Probe Evaluation

Train logistic regression at each layer. Plot accuracy vs. layer.
Identifies the "persona layer" — where traits are most linearly decodable.

Usage:
    python experiments/02_probe.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probe import probe_all_traits, plot_probe_accuracy
import numpy as np
import json

ACTS_DIR   = ROOT / "results" / "activations"
RESULTS    = ROOT / "results"
TRAITS     = ["helpfulness", "sycophancy", "confidence", "verbosity"]


def main():
    results = probe_all_traits(ACTS_DIR, TRAITS, RESULTS)

    if not results:
        print("No probe results computed. Run 01_extract.py first.")
        return

    # Print peak layer summary
    print("\nPeak layers summary:")
    peak_layers = {}
    for trait, accs in results.items():
        peak = int(np.argmax(accs))
        peak_layers[trait] = peak
        print(f"  {trait:15s} layer {peak:2d}  acc={accs[peak]:.3f}")

    (RESULTS / "peak_layers.json").write_text(json.dumps(peak_layers, indent=2))
    print(f"\nSaved peak layers → {RESULTS}/peak_layers.json")

    plot_probe_accuracy(
        results,
        out_path=RESULTS / "figures" / "probe_by_layer.png",
    )
    print("\nDone. Run experiments/03_steer.py next.")


if __name__ == "__main__":
    main()
