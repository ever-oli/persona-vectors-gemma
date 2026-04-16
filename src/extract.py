"""
extract.py — Activation extraction from Gemma 4 via mlx-lm.

Strategy: run a modified forward pass that captures the residual stream
at every transformer layer. Mean-pool over tokens → one vector per layer.

Usage:
    from src.extract import load_model, extract_hidden_states, compute_persona_vector
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str = "mlx-community/gemma-4-e4b-it-4bit"):
    """Load Gemma 4 from HuggingFace Hub (cached after first download ~3GB)."""
    print(f"Loading {model_id} ...")
    model, tokenizer = load(model_id)
    model.eval()
    
    # Gemma 4 has nested structure: model["language_model"]["model"]
    inner_model = model["language_model"]["model"]
    n_layers = len(inner_model["layers"])
    hidden_dim = inner_model["embed_tokens"].weight.shape[-1]
    
    print(f"  Layers: {n_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    return model, tokenizer


def get_inner_model(model):
    """Get the inner language model from Gemma 4 wrapper."""
    return model["language_model"]["model"]


# ---------------------------------------------------------------------------
# Single-prompt activation extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
) -> np.ndarray:
    """
    Run forward pass and collect residual stream at every layer.

    Returns: np.ndarray of shape (n_layers, hidden_dim)
             Mean-pooled over the token dimension.
    """
    tokens = tokenizer.encode(prompt, return_tensors=None)
    input_ids = mx.array([tokens])  # (1, seq_len)

    inner_model = get_inner_model(model)
    layers = inner_model["layers"]
    embed_tokens = inner_model["embed_tokens"]
    n_layers = len(layers)

    # --- embedding ---
    h = embed_tokens(input_ids)  # (1, seq_len, hidden_dim)

    # Gemma scales embeddings by sqrt(hidden_dim)
    hidden_dim = h.shape[-1]
    h = h * (hidden_dim ** 0.5)

    hidden_states = np.zeros((n_layers, hidden_dim), dtype=np.float32)

    # --- layer-by-layer forward ---
    # We build a causal mask manually with same dtype as embeddings (bfloat16)
    seq_len = input_ids.shape[1]
    mask = mx.tril(mx.ones((seq_len, seq_len), dtype=h.dtype))
    mask = mask[None, None, :, :]               # (1, 1, seq_len, seq_len)
    mask = mx.where(mask == 0, mx.array(-1e9, dtype=h.dtype), mx.array(0.0, dtype=h.dtype))

    # Note: shared_kv should NOT be passed between layers in Gemma 4 because
    # different layers have different attention configurations (n_heads varies)
    for i, layer in enumerate(layers):
        # Gemma 4 layer returns: (h, shared_kv, offset)
        # Pass shared_kv=None - each layer computes its own keys/values
        h, _, _ = layer(h, mask=mask, cache=None, shared_kv=None, offset=0)

        # Mean pool over token dimension → (hidden_dim,)
        h_np = np.array(h[0].astype(mx.float32).mean(axis=0))  # (hidden_dim,)
        hidden_states[i] = h_np

    return hidden_states  # (n_layers, hidden_dim)


# ---------------------------------------------------------------------------
# Batch extraction over a JSONL file
# ---------------------------------------------------------------------------

def extract_from_jsonl(
    model,
    tokenizer,
    jsonl_path: Path,
    out_dir: Path,
    trait: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract activations for all pairs in a JSONL file.

    Returns:
        pos_acts: (n_pairs, n_layers, hidden_dim)
        neg_acts: (n_pairs, n_layers, hidden_dim)
    """
    records = [json.loads(l) for l in jsonl_path.read_text().strip().splitlines()]
    n = len(records)
    print(f"Extracting {trait} — {n} pairs ...")

    # Probe first to get shapes
    sample = extract_hidden_states(model, tokenizer, records[0]["positive_prompt"])
    n_layers, hidden_dim = sample.shape

    pos_acts = np.zeros((n, n_layers, hidden_dim), dtype=np.float32)
    neg_acts = np.zeros((n, n_layers, hidden_dim), dtype=np.float32)

    pos_acts[0] = sample
    neg_acts[0] = extract_hidden_states(model, tokenizer, records[0]["negative_prompt"])

    for idx in tqdm(range(1, n), desc=trait):
        pos_acts[idx] = extract_hidden_states(model, tokenizer, records[idx]["positive_prompt"])
        neg_acts[idx] = extract_hidden_states(model, tokenizer, records[idx]["negative_prompt"])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{trait}_positive.npy", pos_acts)
    np.save(out_dir / f"{trait}_negative.npy", neg_acts)
    print(f"  Saved → {out_dir}/{trait}_*.npy  shape={pos_acts.shape}")

    return pos_acts, neg_acts


# ---------------------------------------------------------------------------
# Persona vector computation
# ---------------------------------------------------------------------------

def compute_persona_vector(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
) -> np.ndarray:
    """
    Difference-in-means: mean(positive) - mean(negative), normalized per layer.

    Returns: (n_layers, hidden_dim) — unit-norm direction per layer.
    """
    delta = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)  # (n_layers, hidden_dim)
    norms = np.linalg.norm(delta, axis=1, keepdims=True)   # (n_layers, 1)
    norms = np.where(norms < 1e-8, 1.0, norms)             # avoid div/0
    return delta / norms


def save_persona_vectors(
    vectors: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for trait, vec in vectors.items():
        path = out_dir / f"{trait}_vectors.npy"
        np.save(path, vec)
        print(f"Saved persona vector → {path}  shape={vec.shape}")
