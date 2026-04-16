"""
probe.py — Linear probe: train logistic regression at each layer to classify
positive vs. negative activations. Accuracy by layer reveals where in the
network each trait is encoded.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def probe_trait(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    trait: str,
    cv: int = 5,
) -> np.ndarray:
    """
    Train logistic regression at each layer. Returns accuracy per layer.

    Args:
        pos_acts: (n_pairs, n_layers, hidden_dim)
        neg_acts: (n_pairs, n_layers, hidden_dim)

    Returns: (n_layers,) — mean CV accuracy
    """
    n_pairs, n_layers, hidden_dim = pos_acts.shape

    X = np.concatenate([pos_acts, neg_acts], axis=0)  # (2*n, n_layers, hidden_dim)
    y = np.array([1] * n_pairs + [0] * n_pairs)       # (2*n,)

    accuracies = np.zeros(n_layers)

    for layer in range(n_layers):
        X_layer = X[:, layer, :]  # (2*n, hidden_dim)

        # Standardize (important for logistic regression)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_layer)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        accuracies[layer] = scores.mean()

    print(f"  {trait}: peak layer={np.argmax(accuracies)}, "
          f"peak acc={accuracies.max():.3f}, "
          f"min acc={accuracies.min():.3f}")

    return accuracies


def probe_all_traits(
    acts_dir: Path,
    traits: list[str],
    out_dir: Path,
) -> dict[str, np.ndarray]:
    """Run probe for all traits and save results."""
    results = {}

    for trait in traits:
        pos_path = acts_dir / f"{trait}_positive.npy"
        neg_path = acts_dir / f"{trait}_negative.npy"

        if not pos_path.exists():
            print(f"Skipping {trait} — activations not found at {pos_path}")
            continue

        pos_acts = np.load(pos_path)
        neg_acts = np.load(neg_path)
        print(f"Probing {trait}  shape={pos_acts.shape}")

        accuracies = probe_trait(pos_acts, neg_acts, trait)
        results[trait] = accuracies.tolist()

    # Save JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_accuracy.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved probe results → {out_path}")

    return {t: np.array(v) for t, v in results.items()}


def plot_probe_accuracy(
    results: dict[str, np.ndarray],
    out_path: Path,
    title: str = "Linear Probe Accuracy by Layer — Persona Traits in Gemma 4",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"helpfulness": "#2196F3", "sycophancy": "#F44336",
              "confidence": "#4CAF50", "verbosity": "#FF9800"}

    for trait, accuracies in results.items():
        layers = list(range(len(accuracies)))
        color = colors.get(trait, None)
        ax.plot(layers, accuracies, label=trait, linewidth=2, color=color, marker="o", markersize=3)
        peak_layer = int(np.argmax(accuracies))
        ax.annotate(
            f"{accuracies[peak_layer]:.2f}",
            xy=(peak_layer, accuracies[peak_layer]),
            xytext=(peak_layer + 0.5, accuracies[peak_layer] + 0.01),
            fontsize=8, color=color or "black",
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("5-fold CV Accuracy", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure → {out_path}")
    plt.close(fig)
