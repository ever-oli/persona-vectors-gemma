# Proper Citations for Research Arc

## The Foundational Papers (Chen et al.)

**[1] Persona Vectors (the method)**
- **Citation:** Chen, R., et al. (2025). Persona Vectors: Extracting and Steering Personality Traits in Large Language Models. *arXiv preprint*.
- **What it is:** Original method for extracting contrastive activation directions for behavioral traits
- **Your extension:** Replicated on Gemma 4 via MLX, scaled from 10 to 150 contrastive pairs

**[2] Superficial Alignment (the theory)**
- **Citation:** Chen, R., et al. (2025). Superficial Knowledge in Alignment: A Mechanistic Analysis. *Proceedings of NAACL 2025*.
- **What it is:** Shows alignment can be surface-level (output changes without representation change)
- **Your extension:** Found behavioral traits undergo *deep* restructuring — RLHF rotates representations completely (cos 0.01–0.15), complementing their finding that factual knowledge stays surface-level

**[3] SEAL (the reasoning control)**
- **Citation:** Chen, R., et al. (2025). SEAL: Steerable Reasoning for Language Models. *ICLR 2025*.
- **What it is:** Controls how much models reason via steering vectors
- **Your extension:** Mapped orthogonality between SEAL-style reasoning vectors and persona trait vectors — found reasoning and verbosity are entangled (cos 0.35), while reasoning and helpfulness are orthogonal (cos 0.003)

---

## Your Original Contributions

**[4] This Repository — Four Interconnected Studies**

### Study 1: Replication (Persona Vectors on Gemma 4)
**Reference:** Ortiz, E. (2025). *Persona Vectors in Gemma 4 via MLX: Alignment Taxonomy and Behavioral Geometry* (Study 1). GitHub. https://github.com/ever-oli/persona-vectors-gemma

**What you did:**
- Replicated Chen et al. (2025) Persona Vectors on Gemma 4 4B IT
- Fixed MLX API compatibility issues (nested model structure, layer tuple returns, tied embeddings, logit softcapping)
- Scaled from 10 to 150 contrastive pairs per trait (750 total)
- Found trait peaks: helpfulness (layer 6), sycophancy (layer 0), confidence (layer 2), verbosity (layer 13), reasoning (layer 15)

---

### Study 2: Alignment Depth — Base vs. Instruction-Tuned
**Reference:** Ortiz, E. (2025). *Cross-Model Comparison: Base vs. Instruction-Tuned Gemma 4* (Study 2). In *Persona Vectors in Gemma 4 via MLX*. GitHub.

**What you did:**
- Extracted vectors from both gemma-4-e4b-4bit (base) and gemma-4-e4b-it-4bit (aligned)
- **Finding:** Base model already encodes traits at depth — layers 0, 3, 12, 25
- **Finding:** RLHF relocates them to layers 0, 2, 6, 13 (average shift: 7.8 layers)
- **Finding:** Near-zero cosine similarity (0.01–0.15) indicates RLHF performs *complete rotation* in activation space, not just relocation
- **Implication:** Alignment is representational reorganization, not construction from scratch

**Key sentence:** "RLHF is better at instilling aligned behaviors than removing misaligned ones — helpfulness restructures deeply (cos 0.15) while sycophancy only moderately (cos 0.06), explaining why aligned models remain sycophantic."

---

### Study 3: Artifact Control — Plain-Text Base Model
**Reference:** Ortiz, E. (2025). *Plain-Text Control Study* (Study 3). In *Persona Vectors in Gemma 4 via MLX*. GitHub.

**What you did:**
- Re-ran base model extraction with plain-text prompts (no chat template)
- **Finding:** Chat template does affect peak layers slightly, but fundamental finding holds — base model genuinely has distributed deep representations
- **Methodological contribution:** Controls for template artifacts in base model analysis

---

### Study 4: Behavioral Geometry Map
**Reference:** Ortiz, E. (2025). *Behavioral Geometry: Trait Vector Orthogonality in Gemma 4* (Study 4). In *Persona Vectors in Gemma 4 via MLX*. GitHub.

**What you did:**
- Computed 5×5 cosine similarity matrix at peak layers: helpfulness, sycophancy, confidence, verbosity, reasoning
- **Finding:** Verbosity ↔ Reasoning entangled (cos = +0.35) — step-by-step reasoning produces more words
- **Finding:** Helpfulness ↔ Reasoning orthogonal (cos = +0.003) — safe to compose
- **Finding:** All other pairs orthogonal (|cos| < 0.1)
- **Ordering discovery:** Traits sort by "cognitive load" — sycophancy(0) → confidence(2) → helpfulness(6) → verbosity(13) → reasoning(15)

**Safety implication:** Multi-trait steering is safe for most combinations; only verbosity+reasoning risks interference.

---

## How to Cite in Cover Letters

### For Anthropic (emphasize alignment depth):
> "I replicated Chen et al. (2025) Persona Vectors on Gemma 4 and found that RLHF restructures behavioral trait representations completely (cosine similarity 0.01–0.15 base→IT) rather than creating them from scratch — extending Chen et al. (NAACL 2025) by showing alignment depth varies by trait type."

### For Cohere (emphasize composition):
> "I constructed Gemma 4's first behavioral geometry map (Ortiz, 2025), finding that helpfulness and reasoning vectors are orthogonal (cosine 0.003) and safe to compose, while verbosity and reasoning are entangled (cosine 0.35) — directly applicable to multi-trait steering systems."

### For research positions (full arc):
> "I extended Chen et al. (2025) Persona Vectors through four studies: (1) replication with 150 pairs revealing robust trait encoding at layers 0, 3, 12, 25; (2) showing RLHF relocates and rotates these representations (cos 0.01–0.15); (3) controlling for chat template artifacts; and (4) mapping behavioral geometry to identify safe trait compositions."

---

## BibTeX Entries

```bibtex
% Foundational work
@article{chen2025persona,
  title={Persona Vectors: Extracting and Steering Personality Traits in Large Language Models},
  author={Chen, Runjin and others},
  journal={arXiv preprint},
  year={2025}
}

@inproceedings{chen2025superficial,
  title={Superficial Knowledge in Alignment: A Mechanistic Analysis},
  author={Chen, Runjin and others},
  booktitle={Proceedings of NAACL},
  year={2025}
}

@inproceedings{chen2025seal,
  title={SEAL: Steerable Reasoning for Language Models},
  author={Chen, Runjin and others},
  booktitle={ICLR},
  year={2025}
}

% Your work
@misc{ortiz2025personagemma,
  author = {Ortiz, Ever},
  title = {Persona Vectors in Gemma 4 via MLX: Alignment Taxonomy and Behavioral Geometry},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ever-oli/persona-vectors-gemma}},
  note = {Replication and extension of Chen et al. (2025)}
}
```
