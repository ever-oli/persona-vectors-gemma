# Persona Vectors in Gemma 4

Replication and extension of [Chen et al. (arXiv 2507.21509) — *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*](https://arxiv.org/abs/2507.21509) using **Gemma 4 on Apple Silicon (MLX)**. Four papers, five traits, one M4 Mac Mini.

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

- Chen et al. 2025 — [*Persona Vectors: Monitoring and Controlling Character Traits in Language Models*](https://arxiv.org/abs/2507.21509) (this paper's foundation)
- Chen et al. NAACL 2025 — [*Extracting and Understanding the Superficial Knowledge in Alignment*](https://arxiv.org/abs/2502.04602)
- Chen et al. COLM 2025 — [*SEAL: Steerable Reasoning Calibration of Large Language Models for Free*](https://arxiv.org/abs/2504.07986)
- Zou et al. 2023 — *Representation Engineering*
- Turner et al. 2023 — *Activation Addition*

---

## References

### Foundational Work

1. **Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025).** Persona Vectors: Monitoring and Controlling Character Traits in Language Models. *arXiv:2507.21509*. https://arxiv.org/abs/2507.21509

2. **Chen, R., Perin, G. J., Chen, X., Chen, X., Han, Y., Hirata, N. S. T., Hong, J., & Kailkhura, B. (2025).** Extracting and Understanding the Superficial Knowledge in Alignment. *NAACL 2025*. arXiv:2502.04602. https://arxiv.org/abs/2502.04602

3. **Chen, R., Zhang, Z., Hong, J., Kundu, S., & Wang, Z. (2025).** SEAL: Steerable Reasoning Calibration of Large Language Models for Free. *COLM 2025*. arXiv:2504.07986. https://arxiv.org/abs/2504.07986

### This Repository

4. **Olivares, E. (2025).** *Persona Vectors in Gemma 4 via MLX: Alignment Taxonomy and Behavioral Geometry*. GitHub repository: https://github.com/ever-oli/persona-vectors-gemma

   - **Study 1:** Replication of Chen et al. (2025) Persona Vectors on Gemma 4 (150 contrastive pairs × 5 traits)
   - **Study 2:** Base vs. instruction-tuned Gemma 4 — RLHF relocates and rotates existing trait representations (cosine similarity 0.01–0.15) rather than creating them from scratch
   - **Study 3:** Plain-text control — validates that base model genuinely encodes behavioral traits at depth (layers 0, 3, 12, 25), not a prompt-format artifact
   - **Study 4:** Behavioral geometry map — 5×5 cosine similarity matrix revealing verbosity–reasoning entanglement (cos 0.35) and helpfulness–reasoning orthogonality (cos 0.003)

---

## Citation

```bibtex
@misc{olivares2025personagemma,
  author    = {Olivares, Ever},
  title     = {Persona Vectors in Gemma 4 via {MLX}: Alignment Taxonomy and Behavioral Geometry},
  year      = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ever-oli/persona-vectors-gemma}},
  note      = {Replication and extension of Chen et al.\ arXiv:2507.21509}
}
```

Foundational work:

```bibtex
@article{chen2025persona,
  title   = {Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
  author  = {Chen, Runjin and Arditi, Andy and Sleight, Henry and Evans, Owain and Lindsey, Jack},
  journal = {arXiv preprint arXiv:2507.21509},
  year    = {2025}
}

@inproceedings{chen2025superficial,
  title     = {Extracting and Understanding the Superficial Knowledge in Alignment},
  author    = {Chen, Runjin and Perin, Gabriel Jacob and Chen, Xuxi and Chen, Xilun and Han, Yan and Hirata, Nina S. T. and Hong, Junyuan and Kailkhura, Bhavya},
  booktitle = {Proceedings of NAACL 2025},
  year      = {2025}
}

@inproceedings{chen2025seal,
  title     = {{SEAL}: Steerable Reasoning Calibration of Large Language Models for Free},
  author    = {Chen, Runjin and Zhang, Zhenyu and Hong, Junyuan and Kundu, Souvik and Wang, Zhangyang},
  booktitle = {Proceedings of COLM 2025},
  year      = {2025}
}
```
