# Persona Vectors in Gemma 4 — Research Plan

**Goal**: Reproduce and extend Runjin Chen's *Persona Vectors* (arXiv 2025) using Gemma 4
on Apple Silicon (M4 Mac Mini 16GB). Novel contribution: test vector composability.

**Connection**: Runjin Chen (UT Austin VITA → Anthropic MTS) is the original author.
This directly engages her published alignment research on a modern model she likely uses.

---

## The Core Idea (Paper 10 Summary)

Persona Vectors are directions in an LLM's residual stream that encode behavioral traits
(helpfulness, sycophancy, confidence, etc.). Extract the direction, probe it, steer with it.

Three claims in the original paper:
1. Traits have a **linear representation** in activation space
2. A **single direction** per layer captures most of the variance
3. Adding/subtracting the vector **steers behavior** during generation

Our extension — Claim 4 (novel):
4. Multiple trait vectors **compose additively**: `steer(helpful) + steer(confident)` ≈
   `steer(helpful+confident)`. This tests the linear representation hypothesis harder.

---

## Model

Primary: `mlx-community/gemma-4-e4b-it-4bit`
Fallback: `unsloth/gemma-4-E4B-it-UD-MLX-4bit`

Gemma 4 E4B = ~4B effective params, 4-bit quantized ≈ 2-3GB VRAM.
Runs in ~4-6GB on M4 unified memory. Plenty of headroom.

**Why Gemma 4 specifically:**
- Instruction-tuned → behavioral traits are already present (not latent)
- Google's current best small model → relevant to industry
- MLX-native quantization → fast on M4
- Not a model Runjin's paper tested → genuinely new data point

---

## Traits to Study (4 total)

| Trait | Positive condition | Negative condition | Why interesting |
|-------|-------------------|--------------------|-----------------|
| **Helpfulness** | Thorough, complete answers | Minimal, dismissive answers | Core alignment trait |
| **Sycophancy** | Agree with wrong user claim | Correct the user | Known alignment failure mode |
| **Confidence** | Assertive, direct statements | Hedged, uncertain statements | Calibration-relevant |
| **Verbosity** | Long detailed responses | Short terse responses | Control/sanity check |

Verbosity is the "sanity check" — it should have a strong, easily detectable vector.
If verbosity doesn't work, the pipeline is broken. Helpfulness + sycophancy are the
scientifically interesting ones.

---

## 5-Phase Plan

### Phase 0: Environment (Day 1)
```
pip install mlx-lm scikit-learn matplotlib tqdm jsonlines
mlx_lm.convert --help   # verify install
python -c "from mlx_lm import load; m,t = load('mlx-community/gemma-4-e4b-it-4bit')"
```

Deliverable: model loads, generates text, no OOM.

---

### Phase 1: Dataset Construction (Days 2-3)

Build 200 contrastive prompt pairs per trait. Format: JSONL.

Each record:
```json
{
  "trait": "helpfulness",
  "positive_prompt": "User: How do I sort a list in Python?\nAssistant:",
  "negative_prompt": "User: How do I sort a list in Python?\nAssistant: Look it up.",
  "source": "synthetic"
}
```

**Construction strategy:**
- 100 pairs: templates (varied topics, same structural contrast)
- 100 pairs: few-shot steered generations (use the model itself to generate both sides)

Files:
- `data/prompts/helpfulness.jsonl`
- `data/prompts/sycophancy.jsonl`
- `data/prompts/confidence.jsonl`
- `data/prompts/verbosity.jsonl`

Script: `src/dataset.py` — generates all four files.

---

### Phase 2: Activation Extraction (Days 4-6)

For each prompt in a pair, run a forward pass and collect the **residual stream** at every
transformer layer. Mean-pool over the token dimension → one vector per layer per prompt.

```
prompt → tokenize → model forward (layer 0..N) → collect h[i] at each layer → mean pool → save
```

Output: `results/activations/{trait}_positive.npy`, `results/activations/{trait}_negative.npy`
Shape: `(n_pairs, n_layers, hidden_dim)` = `(200, ~28, 2304)` for Gemma 4 E4B

**Persona vector extraction:**
```python
# For each layer i:
pos_mean = activations_positive[:, i, :].mean(axis=0)  # (hidden_dim,)
neg_mean = activations_negative[:, i, :].mean(axis=0)
direction = pos_mean - neg_mean
direction = direction / np.linalg.norm(direction)  # normalize
persona_vectors[trait][layer] = direction
```

Save: `results/vectors/{trait}_vectors.npy` — shape `(n_layers, hidden_dim)`

Script: `experiments/01_extract.py`

---

### Phase 3: Linear Probe Validation (Days 7-8)

Train a logistic regression at each layer to classify positive vs. negative activations.
Accuracy by layer tells you *where* in the network the trait lives.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

for layer in range(n_layers):
    X = activations[:, layer, :]  # (400, hidden_dim) — 200 pos + 200 neg
    y = labels                     # (400,) — 0 or 1
    scores = cross_val_score(LogisticRegression(), X, y, cv=5)
    probe_accuracy[trait][layer] = scores.mean()
```

**Expected result**: accuracy peaks at middle-to-late layers (~60-80% through the network).
This is the "persona layer" — use it for steering.

Visualize: line plot, accuracy vs. layer, one line per trait.

Script: `experiments/02_probe.py`
Output: `results/probe_accuracy.json`, `results/figures/probe_by_layer.png`

---

### Phase 4: Activation Steering (Days 9-11)

Inject the persona vector into the model's residual stream during generation.

```python
alpha = 15.0  # steering strength (tune this)
target_layer = peak_layer[trait]

# During forward pass at target_layer:
h[target_layer] += alpha * persona_vectors[trait][target_layer]
```

For MLX, implement this as a modified forward pass (no PyTorch-style hooks).
See `src/steer.py` for the wrapper.

**Evaluation — three prompts per trait, show before/after:**

| Trait | Prompt | Before | After (+vec) | After (-vec) |
|-------|--------|--------|--------------|--------------|
| Helpfulness | "Explain recursion" | baseline | more thorough | dismissive |
| Sycophancy | "2+2=5, right?" | baseline | "Yes, you're right" | "No, 2+2=4" |
| Confidence | "What is the capital of France?" | baseline | "Paris." | "I think maybe Paris?" |

Save qualitative examples as markdown in `results/steering_examples.md`.

Script: `experiments/03_steer.py`

---

### Phase 5: Composability Experiment — Novel Contribution (Days 12-14)

**Hypothesis**: Persona vectors compose additively in activation space.

Test: steer with `v_helpful + v_confident` simultaneously. Does the model become both
more helpful AND more confident? Or does one trait dominate?

Three conditions:
```
A: steer with helpful alone      (alpha=15)
B: steer with confident alone    (alpha=15)
C: steer with helpful+confident  (alpha=15 each, applied simultaneously at same layer)
```

**Metrics:**
- Helpfulness proxy: response length, question coverage score (simple keyword matching)
- Confidence proxy: hedge word count ("maybe", "perhaps", "I think", "might") — lower = more confident
- Sycophancy: agreement rate with intentionally wrong user statements

**Composability score:**
```
compose_score = corr(A_metric + B_metric, C_metric)
```
If close to 1.0 → vectors compose linearly (supports superposition hypothesis).
If much less → non-linear interaction (interesting finding either way).

Script: `experiments/04_compose.py`
Output: `results/composability.json`, `results/figures/composability_heatmap.png`

---

## Writeup Structure (targeting arXiv preprint)

```
1. Introduction
   - Why controllable alignment behavior matters
   - Prior work: Representation Engineering, CAA, Persona Vectors (Chen et al. 2025)
   - Our contribution: first composability test on Gemma 4

2. Method
   2.1 Contrastive pair construction
   2.2 Activation extraction (residual stream, mean pooling)
   2.3 Persona vector computation (difference-in-means)
   2.4 Linear probe evaluation
   2.5 Activation steering

3. Results
   3.1 Probe accuracy by layer (Figure 1)
   3.2 Qualitative steering examples (Table 1)
   3.3 Composability experiment (Figure 2 + Table 2)

4. Discussion
   - Where in Gemma 4 do traits live?
   - Does additive composability hold?
   - Failure cases (what doesn't steer well?)

5. Conclusion + Future Work
   - Cross-model transfer (Gemma → Llama)
   - Fine-tuning stability (do persona vectors survive RLHF?)
```

---

## M4 16GB Budget

| Operation | Memory | Time estimate |
|-----------|--------|---------------|
| Model load (4-bit) | ~3GB | 30s |
| Activation extraction (200 pairs × 4 traits) | peaks at ~6GB | ~45 min |
| Linear probe training | <1GB | ~5 min |
| Steering generation (qualitative) | ~3GB | ~10 min |
| **Total peak** | **~6-7GB** | **~1 hr/run** |

Safe. 16GB unified memory has plenty of headroom for OS + Python overhead.

---

## File Map

```
persona-vectors-gemma/
├── RESEARCH_PLAN.md          ← this file
├── requirements.txt
├── src/
│   ├── dataset.py            # generate contrastive prompt pairs
│   ├── extract.py            # activation extraction from mlx-lm model
│   ├── probe.py              # linear probe training + evaluation
│   └── steer.py              # activation steering wrapper
├── experiments/
│   ├── 01_extract.py         # run extraction pipeline
│   ├── 02_probe.py           # run probing pipeline
│   ├── 03_steer.py           # run steering experiments
│   └── 04_compose.py         # composability experiment
├── data/
│   └── prompts/
│       ├── helpfulness.jsonl
│       ├── sycophancy.jsonl
│       ├── confidence.jsonl
│       └── verbosity.jsonl
├── results/
│   ├── activations/          # .npy files (gitignored — large)
│   ├── vectors/              # persona vectors per trait
│   ├── figures/              # plots
│   └── steering_examples.md  # qualitative results
└── notebooks/
    └── visualization.ipynb   # figures for paper
```

---

## Connection to Job Applications

This project directly addresses:

| Role | How this project helps |
|------|------------------------|
| Anthropic (any) | "I replicated + extended Runjin Chen's Persona Vectors work on Gemma 4, found [result] about composability" |
| Arize AI PM | "I built an interpretability pipeline and measured LLM behavioral trait detection" |
| Cohere AAE | "I have hands-on experience with activation steering and LLM behavioral control" |
| Any alignment role | First-author-quality work on a live alignment problem |

Mention in cover letter: "While preparing this application I implemented activation-based
persona vector extraction on Gemma 4 (MLX), extending Chen et al. 2025 with a composability
analysis. Code at github.com/[your-handle]/persona-vectors-gemma."
