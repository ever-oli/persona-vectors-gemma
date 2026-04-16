# Persona Vectors in Gemma 4

Replication and extension of [Chen et al. (arXiv 2025) — *Persona Vectors: Monitoring Character Traits in LLMs*](https://arxiv.org/abs/2502.12024) using **Gemma 4 on Apple Silicon (MLX)**. Four papers, five traits, one M4 Mac Mini.

---

## Key Findings

### 1. Traits encode in a reactive-to-cognitive ordering (Paper 10)

```
sycophancy(layer 0) → confidence(2) → helpfulness(6) → verbosity(13) → reasoning(15)
```

Social/reflexive traits encode earliest. Cognitive/generative traits encode latest. This ordering is stable across 150 contrastive pairs.

### 2. RLHF relocates representations, not creates them (Papers 12–13)

Base Gemma 4 already encodes all five traits at depth (layers 0, 3, 12, 25). After instruction tuning, traits shift to layers 0, 2, 6, 13 — average shift of 7.8 layers. Base→IT cosine similarity: **0.01–0.15** (near-orthogonal directions).

RLHF is a relocation + rotation operation on pre-existing representations, not construction from scratch.

### 3. Underpowered extraction gives qualitatively wrong results (methodological)

At 10 pairs/trait, the base model appeared to have no deep representations (everything at layer 0). At 150 pairs, the distributed structure emerged. Representation extraction studies with <50 pairs per trait should be treated with caution.

### 4. Verbosity–reasoning entanglement constrains multi-trait steering (Paper 14)

| Pair | Cosine | Safe to compose? |
|------|--------|-----------------|
| helpfulness ↔ reasoning | 0.003 | ✅ Orthogonal |
| sycophancy ↔ reasoning | -0.014 | ✅ Orthogonal |
| confidence ↔ reasoning | -0.002 | ✅ Orthogonal |
| **verbosity ↔ reasoning** | **0.35** | ⚠️ Entangled |
| helpfulness ↔ sycophancy | -0.030 | ✅ Orthogonal |

Step-by-step reasoning and verbose output share representational space. Steering one will partially steer the other.

---

## Papers

| Paper | Question | Method |
|-------|----------|--------|
| **10** | Do persona vectors exist in Gemma 4? | Contrastive activation extraction, linear probes |
| **12** | Does RLHF change the vectors? | Base vs. IT comparison (chat-template prompts) |
| **13** | Is that an artifact of the prompt format? | Base model re-run with plain-text prompts |
| **14** | Can traits be composed safely? | 5×5 cosine similarity matrix at peak layers |

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate prompt pairs (150 per trait)
python src/dataset.py

# Full pipeline
python experiments/01_extract.py   # ~45 min on M4
python experiments/02_probe.py
python experiments/03_steer.py
python experiments/04b_compose_fixed.py
python experiments/05_cross_model.py   # base vs IT
python experiments/06_paper13.py       # plain-text control
python experiments/07_paper14.py       # behavioral geometry
```

---

## Model

`mlx-community/gemma-4-e4b-it-4bit` — ~3GB, runs in ~5GB peak on M4 unified memory.

---

## Results

```
results/
├── vectors/                  # 5 persona vectors (42 × 2560) from Gemma 4-it
├── vectors_base/             # Gemma 4 base (chat-template prompts)
├── vectors_base_plain/       # Gemma 4 base (plain-text prompts)
├── probe_accuracy.json       # Layer-wise accuracy for all 5 traits
├── peak_layers.json          # Peak layer per trait
├── alignment_effect.json     # Paper 12: base vs IT
├── alignment_taxonomy.json   # Paper 13: plain-text control
├── behavioral_geometry.json  # Paper 14: 5×5 cosine matrix
└── figures/
    ├── probe_by_layer.png
    ├── base_vs_it_probes.png
    ├── alignment_taxonomy.png
    └── behavioral_geometry.png
```

---

## Limitations

- 150 contrastive pairs per trait (sycophancy: 272). Results are directionally robust but confidence intervals are wide.
- Single model family (Gemma 4 E4B). Cross-architecture generalization untested.
- Proxy metrics for steering evaluation (word count, hedge-word rate). Not human-evaluated.
- Plain-text base model saw instruction-style prompts it wasn't trained to follow.

---

## Related Work

- Chen et al. 2025 — *Persona Vectors* (this paper's foundation)
- Chen et al. NAACL 2025 — *Superficial Knowledge in Alignment*
- Zou et al. 2023 — *Representation Engineering*
- Turner et al. 2023 — *Activation Addition*
