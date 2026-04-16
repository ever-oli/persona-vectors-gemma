"""
steer.py — Activation steering for Gemma 4 via MLX.

Inject a persona vector into the residual stream at a target layer
during text generation. Alpha > 0 amplifies the trait; alpha < 0 suppresses it.

Usage:
    from src.steer import generate_steered

    text = generate_steered(
        model, tokenizer, prompt,
        persona_vector=vectors["helpfulness"],
        target_layer=14,
        alpha=15.0,
        max_tokens=200,
    )
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _get_inner_model(model):
    """Get the inner language model from Gemma 4 wrapper."""
    return model["language_model"]["model"]


def _get_lm(model):
    """Get the language model wrapper (has tie_word_embeddings, etc)."""
    return model["language_model"]


def _compute_logits(lm, h_norm):
    """Compute logits from normalized hidden states."""
    # Gemma 4 uses tied word embeddings
    if lm.tie_word_embeddings:
        logits = lm.model.embed_tokens.as_linear(h_norm)
    else:
        logits = lm.lm_head(h_norm)
    
    # Apply logit softcapping if present
    if lm.final_logit_softcapping is not None:
        logits = mx.tanh(logits / lm.final_logit_softcapping) * lm.final_logit_softcapping
    
    return logits


# ---------------------------------------------------------------------------
# Steered generation
# ---------------------------------------------------------------------------

def generate_steered(
    model,
    tokenizer,
    prompt: str,
    persona_vector: np.ndarray,
    target_layer: int,
    alpha: float = 15.0,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate text with activation steering at target_layer.

    persona_vector: (n_layers, hidden_dim) — only target_layer slice is used.
    alpha: steering strength. Positive = toward positive condition. Negative = away.
    """
    # Extract the direction for the target layer and convert to MLX
    direction = mx.array(persona_vector[target_layer])          # (hidden_dim,)
    direction = direction / (mx.linalg.norm(direction) + 1e-8)  # unit norm

    tokens = tokenizer.encode(prompt, return_tensors=None)
    input_ids = mx.array([tokens])  # (1, seq_len)

    inner_model = _get_inner_model(model)
    lm = _get_lm(model)
    layers = inner_model["layers"]
    n_layers = len(layers)

    # Initial prompt forward pass
    h = inner_model["embed_tokens"](input_ids)
    hidden_dim = h.shape[-1]
    h = h * (hidden_dim ** 0.5)

    seq_len = input_ids.shape[1]
    mask = mx.tril(mx.ones((seq_len, seq_len), dtype=h.dtype))[None, None]
    mask = mx.where(mask == 0, mx.array(-1e9, dtype=h.dtype), mx.array(0.0, dtype=h.dtype))

    # Note: shared_kv should NOT be passed between layers in Gemma 4 because
    # different layers have different attention configurations (n_heads varies)
    for i, layer in enumerate(layers):
        h, _, _ = layer(h, mask=mask, cache=None, shared_kv=None, offset=0)
        if i == target_layer:
            # Inject: add alpha * direction to every token position
            h = h + alpha * direction[None, None, :]

    # Apply final norm and get logits for first token
    h_norm = inner_model["norm"](h)
    logits = _compute_logits(lm, h_norm)

    generated = list(tokens)

    for _ in range(max_tokens):
        # Sample from logits at last position
        last_logits = logits[0, -1, :]  # (vocab,)
        next_token = _sample(last_logits, temperature=temperature, top_p=top_p)
        generated.append(int(next_token))

        # Check for EOS
        if int(next_token) == tokenizer.eos_token_id:
            break

        # Single-token forward pass with KV cache
        next_input = mx.array([[int(next_token)]])
        h = inner_model["embed_tokens"](next_input)
        h = h * (hidden_dim ** 0.5)

        # For single-token generation, no mask needed (only attends to itself)
        for i, layer in enumerate(layers):
            h, _, _ = layer(h, mask=None, cache=None, shared_kv=None, offset=0)
            if i == target_layer:
                h = h + alpha * direction[None, None, :]

        h_norm = inner_model["norm"](h)
        logits = _compute_logits(lm, h_norm)

    # Decode only the new tokens (after the prompt)
    new_tokens = generated[len(tokens):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _sample(logits: mx.array, temperature: float = 1.0, top_p: float = 0.9) -> mx.array:
    """Top-p (nucleus) sampling."""
    if temperature == 0:
        return mx.argmax(logits)

    logits = logits / temperature
    probs = mx.softmax(logits)

    # Convert to numpy for top-p filtering (simpler)
    # Cast to float32 first to avoid bfloat16 numpy issues
    probs_np = np.array(probs.astype(mx.float32))
    sorted_idx = np.argsort(-probs_np)
    sorted_probs = probs_np[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, top_p) + 1
    sorted_probs[cutoff:] = 0.0
    sorted_probs /= sorted_probs.sum()

    token_id = np.random.choice(sorted_idx, p=sorted_probs)
    return mx.array(token_id)


# ---------------------------------------------------------------------------
# Composability: apply multiple vectors simultaneously
# ---------------------------------------------------------------------------

def generate_composed(
    model,
    tokenizer,
    prompt: str,
    persona_vectors: dict[str, np.ndarray],
    alphas: dict[str, float],
    target_layer: int,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """
    Apply multiple persona vectors simultaneously at target_layer.

    persona_vectors: {"helpfulness": array, "confidence": array, ...}
    alphas:          {"helpfulness": 15.0, "confidence": 10.0, ...}
    """
    # Sum the scaled directions
    combined = None
    for trait, vec in persona_vectors.items():
        alpha = alphas.get(trait, 15.0)
        direction = mx.array(vec[target_layer])
        direction = direction / (mx.linalg.norm(direction) + 1e-8)
        scaled = alpha * direction
        combined = scaled if combined is None else combined + scaled

    if combined is None:
        raise ValueError("No persona vectors provided.")

    # Convert to single-vector numpy array and call generate_steered with alpha=1
    # We embed the combined vector directly as the "direction" at alpha=1
    combined_np = np.array(combined)

    # Patch: temporarily replace the target layer's direction
    # Easiest: create a fake "vectors" array where target_layer = combined_np
    inner_model = _get_inner_model(model)
    n_layers = len(inner_model["layers"])
    hidden_dim = combined_np.shape[0]
    fake_vectors = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    fake_vectors[target_layer] = combined_np

    return generate_steered(
        model, tokenizer, prompt,
        persona_vector=fake_vectors,
        target_layer=target_layer,
        alpha=1.0,  # already scaled above
        max_tokens=max_tokens,
        temperature=temperature,
    )
