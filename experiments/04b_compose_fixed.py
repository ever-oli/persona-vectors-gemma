#!/usr/bin/env python3
"""
Paper 10 — Composability (fixed): use each vector at its own peak layer.

Paper 10's original composability test (04_compose.py) applied BOTH vectors at
layer 0 (helpfulness peak). Confidence peaks at layer 3, so the confidence
dimension showed 0.0 for all conditions — a layer-mismatch artifact, not a real result.

This version:
  - Applies helpfulness at layer 0 (its peak)
  - Applies confidence at layer 3 (its peak)
  - For the composed condition, applies both simultaneously at their respective layers

This tests multi-layer composition, which is a harder and more interesting question:
  "Do traits that live at different layers still compose when applied jointly?"
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import mlx.core as mx
from src.extract import load_model
from src.steer import _get_inner_model, _get_lm, _sample, _compute_logits

VECTORS_DIR = ROOT / "results" / "vectors"
RESULTS     = ROOT / "results"
DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"

COMPOSE_PROMPTS = [
    "<start_of_turn>user\nShould I use microservices or a monolith for my startup?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat's the most important thing to learn in machine learning?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nIs remote work better than in-office for software engineers?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat database should I use for a new web app?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nHow do I get better at coding?\n<end_of_turn>\n<start_of_turn>model\n",
]

HEDGE_WORDS = {"maybe", "perhaps", "possibly", "might", "could", "probably",
               "i think", "i believe", "not sure", "uncertain", "it depends"}


def helpfulness_score(t): return len(t.split())
def confidence_score(t): return -sum(1 for w in HEDGE_WORDS if w in t.lower())


def generate_multi_layer(
    model, tokenizer, prompt: str,
    injections: list[tuple[int, np.ndarray, float]],   # [(layer, vector, alpha), ...]
    max_tokens: int = 120,
    temperature: float = 0.7,
) -> str:
    """Generate with steering injected at multiple (potentially different) layers."""
    # Pre-compute direction tensors
    directions = [
        (layer, mx.array(vec / (np.linalg.norm(vec) + 1e-8)), alpha)
        for layer, vec, alpha in injections
    ]

    tokens = tokenizer.encode(prompt, return_tensors=None)
    input_ids = mx.array([tokens])

    inner_model = _get_inner_model(model)
    lm = _get_lm(model)
    layers_list = inner_model["layers"]
    n_layers = len(layers_list)

    h = inner_model["embed_tokens"](input_ids)
    hidden_dim = h.shape[-1]
    h = h * (hidden_dim ** 0.5)

    seq_len = input_ids.shape[1]
    mask = mx.tril(mx.ones((seq_len, seq_len), dtype=h.dtype))[None, None]
    mask = mx.where(mask == 0, mx.array(-1e9, dtype=h.dtype), mx.array(0.0, dtype=h.dtype))

    inject_map = {layer: (d, a) for layer, d, a in directions}

    for i, layer in enumerate(layers_list):
        h, _, _ = layer(h, mask=mask, cache=None, shared_kv=None, offset=0)
        if i in inject_map:
            d, a = inject_map[i]
            h = h + a * d[None, None, :]

    h_norm = inner_model["norm"](h)
    logits = _compute_logits(lm, h_norm)

    generated = list(tokens)
    for _ in range(max_tokens):
        next_token = _sample(logits[0, -1, :], temperature=temperature)
        generated.append(int(next_token))
        if int(next_token) == tokenizer.eos_token_id:
            break

        next_input = mx.array([[int(next_token)]])
        h = inner_model["embed_tokens"](next_input)
        h = h * (hidden_dim ** 0.5)

        for i, layer in enumerate(layers_list):
            h, _, _ = layer(h, mask=None, cache=None, shared_kv=None, offset=0)
            if i in inject_map:
                d, a = inject_map[i]
                h = h + a * d[None, None, :]

        h_norm = inner_model["norm"](h)
        logits = _compute_logits(lm, h_norm)

    return tokenizer.decode(generated[len(tokens):], skip_special_tokens=True)


def main():
    peak_layers = json.loads((RESULTS / "peak_layers.json").read_text())

    h_layer = peak_layers["helpfulness"]   # 0
    c_layer = peak_layers["confidence"]    # 3
    alpha   = 12.0

    h_vec = np.load(VECTORS_DIR / "helpfulness_vectors.npy")[h_layer]
    c_vec = np.load(VECTORS_DIR / "confidence_vectors.npy")[c_layer]

    model, tokenizer = load_model()

    rows = []
    for i, prompt in enumerate(COMPOSE_PROMPTS):
        print(f"\nPrompt {i+1}: {prompt.strip()[:60]}...")

        baseline = generate_multi_layer(model, tokenizer, prompt, [])
        h_only   = generate_multi_layer(model, tokenizer, prompt, [(h_layer, h_vec, alpha)])
        c_only   = generate_multi_layer(model, tokenizer, prompt, [(c_layer, c_vec, alpha)])
        composed = generate_multi_layer(model, tokenizer, prompt, [
            (h_layer, h_vec, alpha), (c_layer, c_vec, alpha)
        ])

        rows.append({
            "baseline":  {"text": baseline,  "help": helpfulness_score(baseline),  "conf": confidence_score(baseline)},
            "h_only":    {"text": h_only,     "help": helpfulness_score(h_only),    "conf": confidence_score(h_only)},
            "c_only":    {"text": c_only,     "help": helpfulness_score(c_only),    "conf": confidence_score(c_only)},
            "composed":  {"text": composed,   "help": helpfulness_score(composed),  "conf": confidence_score(composed)},
        })

    def avg(cond, metric):
        return np.mean([r[cond][metric] for r in rows])

    results = {}
    for metric in ["help", "conf"]:
        base = avg("baseline", metric)
        A    = avg("h_only",   metric)
        B    = avg("c_only",   metric)
        C    = avg("composed", metric)
        pred = A + B - base
        score = 1 - abs(C - pred) / (abs(pred - base) + 1e-6)

        results[metric] = {
            "baseline": round(base, 2), "h_only": round(A, 2),
            "c_only": round(B, 2),      "composed": round(C, 2),
            "predicted_additive": round(pred, 2),
            "composability_score": round(float(score), 3),
        }
        print(f"\n{metric}: baseline={base:.1f}, h_only={A:.1f}, c_only={B:.1f}, "
              f"composed={C:.1f}, predicted={pred:.1f}, score={score:.3f}")

    out = RESULTS / "composability_fixed.json"
    out.write_text(json.dumps({
        "results": results,
        "note": "Fixed: each vector applied at its own peak layer (h@0, c@3)",
        "h_layer": h_layer, "c_layer": c_layer, "alpha": alpha,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
