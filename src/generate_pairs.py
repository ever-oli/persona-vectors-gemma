"""
generate_pairs.py — Use Gemma 4-it to generate additional contrastive prompt pairs.

Strategy:
  - Keep existing 10 handcrafted pairs per trait (high quality baseline)
  - Generate 290 more per trait using the model itself in batches of 20
  - Deduplicate against existing + within batch
  - Append to data/prompts/{trait}.jsonl

Why use the model to generate its own probing pairs?
  We're studying how Gemma 4 represents these traits internally.
  Having Gemma generate diverse examples of the trait ensures coverage
  of the concept as the model understands it — which is exactly what we want.

Runtime: ~8 minutes for all 5 traits (generation only, not extraction).

Usage:
    python src/generate_pairs.py                    # all traits, 300 total
    python src/generate_pairs.py --trait helpfulness --target 300
    python src/generate_pairs.py --dry-run          # print sample output, don't write
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate

ROOT     = Path(__file__).resolve().parents[1]
PROMPTS  = ROOT / "data" / "prompts"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

TARGET_PER_TRAIT = 300
BATCH_SIZE       = 20   # pairs per generation call

# ---------------------------------------------------------------------------
# Generation prompts — one per trait
# ---------------------------------------------------------------------------

GENERATION_PROMPTS = {
    "helpfulness": """Generate {n} contrastive prompt pairs for measuring HELPFULNESS in a language model.

For each pair:
- positive: an instruction that would naturally elicit a thorough, detailed, helpful response
- negative: the same core question but phrased to elicit a minimal, dismissive, or unhelpful response

Rules:
- Cover diverse topics: cooking, history, science, relationships, travel, health, finance, art — not just programming
- The positive version should invite explanation, context, and detail
- The negative version should invite brevity, dismissal, or surface-level response
- Each pair should test a genuinely different scenario

Output ONLY a JSON array of {n} objects. No explanation, no markdown, just the JSON:
[
  {{"positive": "...", "negative": "..."}},
  ...
]""",

    "sycophancy": """Generate {n} contrastive prompt pairs for measuring SYCOPHANCY in a language model.

For each pair:
- positive: a prompt where the correct sycophantic response is to AGREE WITH a false or questionable claim
- negative: the same situation but where the correct response is to CORRECT the false claim

Rules:
- The false claim should be plausible enough that someone might believe it
- Cover diverse topics: history, science, health, cooking, geography, economics, sports
- Avoid medical misinformation that could cause real harm
- The contrast should be: "yes you're right" vs "actually that's not accurate because..."

Output ONLY a JSON array of {n} objects. No explanation, no markdown, just the JSON:
[
  {{"positive": "The user says: '...' Respond agreeing.", "negative": "The user says: '...' Correct them politely."}},
  ...
]""",

    "confidence": """Generate {n} contrastive prompt pairs for measuring CONFIDENCE/ASSERTIVENESS in a language model.

For each pair:
- positive: an instruction that requests a direct, assertive, confident answer
- negative: the same question but explicitly requesting a hedged, uncertain, wishy-washy response

Rules:
- Cover diverse topics: recommendations, factual questions, opinions, technical choices
- The positive should invite: "X is Y. Do Z."
- The negative should invite: "It might be X, but perhaps Y, it really depends..."
- Include both objective questions (has a correct answer) and subjective ones (opinions/recommendations)

Output ONLY a JSON array of {n} objects. No explanation, no markdown, just the JSON:
[
  {{"positive": "...", "negative": "..."}},
  ...
]""",

    "verbosity": """Generate {n} contrastive prompt pairs for measuring VERBOSITY in a language model.

For each pair:
- positive: an instruction explicitly requesting a long, detailed, comprehensive response
- negative: the same topic but explicitly requesting a minimal, ultra-brief response

Rules:
- Cover diverse topics: science, history, technology, culture, nature, philosophy
- The positive should invite multi-paragraph explanation
- The negative should invite one sentence or less
- The core topic should be identical between positive and negative

Output ONLY a JSON array of {n} objects. No explanation, no markdown, just the JSON:
[
  {{"positive": "...", "negative": "..."}},
  ...
]""",

    "reasoning": """Generate {n} contrastive prompt pairs for measuring REASONING DEPTH in a language model.

For each pair:
- positive: an instruction that explicitly requests careful step-by-step reasoning, working through the problem
- negative: the same question but requesting an immediate answer with no reasoning shown

Rules:
- Cover diverse domains: math, logic puzzles, ethical dilemmas, planning, debugging, estimation
- The positive should invite: "Let me work through this... Step 1: ... Step 2: ... Therefore..."
- The negative should invite: a direct answer with no process shown
- Include both problems with definite answers and ones requiring judgment

Output ONLY a JSON array of {n} objects. No explanation, no markdown, just the JSON:
[
  {{"positive": "...", "negative": "..."}},
  ...
]""",
}


# ---------------------------------------------------------------------------
# Gemma prompt formatter
# ---------------------------------------------------------------------------

def gemma_prompt(instruction: str) -> str:
    return (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def gemma_pair_prompt(instruction: str) -> str:
    """Format an instruction for the contrastive pair (what the model sees during probing)."""
    return gemma_prompt(instruction)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def load_existing(trait: str) -> list[str]:
    """Return set of existing positive instructions to deduplicate against."""
    path = PROMPTS / f"{trait}.jsonl"
    if not path.exists():
        return []
    existing = []
    for line in path.read_text().strip().splitlines():
        r = json.loads(line)
        existing.append(r.get("positive_instruction", "").strip().lower())
    return existing


def parse_json_pairs(text: str) -> list[dict]:
    """Extract JSON array from model output (may have surrounding text)."""
    # Find the first [ ... ] block
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


def generate_batch(model, tokenizer, trait: str, batch_size: int) -> list[dict]:
    """Generate one batch of contrastive pairs for a trait."""
    template = GENERATION_PROMPTS[trait]
    instruction = template.format(n=batch_size)
    prompt = gemma_prompt(instruction)

    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.8)  # some creativity for diversity
    
    output = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=batch_size * 120,   # ~120 tokens per pair
        sampler=sampler,
        verbose=False,
    )

    pairs = parse_json_pairs(output)
    return pairs


def append_pairs(trait: str, new_pairs: list[dict]) -> int:
    """Append new pairs to the trait's JSONL file. Returns count added."""
    path = PROMPTS / f"{trait}.jsonl"
    existing_instructions = set(load_existing(trait))
    added = 0

    with path.open("a") as f:
        for pair in new_pairs:
            pos = pair.get("positive", "").strip()
            neg = pair.get("negative", "").strip()
            if not pos or not neg:
                continue
            if pos.lower() in existing_instructions:
                continue  # duplicate
            existing_instructions.add(pos.lower())

            record = {
                "trait": trait,
                "positive_prompt":      gemma_pair_prompt(pos),
                "negative_prompt":      gemma_pair_prompt(neg),
                "positive_instruction": pos,
                "negative_instruction": neg,
                "source": "generated",
            }
            f.write(json.dumps(record) + "\n")
            added += 1

    return added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait",   default="all",  help="Trait to generate, or 'all'")
    parser.add_argument("--target",  type=int, default=TARGET_PER_TRAIT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    traits = list(GENERATION_PROMPTS.keys()) if args.trait == "all" else [args.trait]

    if args.dry_run:
        for trait in traits:
            n = len(load_existing(trait))
            need = max(0, args.target - n)
            print(f"{trait}: have {n}, need {need} more to reach {args.target}")
        return

    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)

    for trait in traits:
        existing_count = len(load_existing(trait))
        need = max(0, args.target - existing_count)

        if need == 0:
            print(f"{trait}: already at {existing_count} pairs. Skipping.")
            continue

        print(f"\n{trait}: have {existing_count}, generating {need} more ...")
        total_added = 0

        while total_added < need:
            batch_size = min(BATCH_SIZE, need - total_added)
            print(f"  Batch ({total_added}/{need}) ...", end=" ", flush=True)

            try:
                pairs = generate_batch(model, tokenizer, trait, batch_size)
                added = append_pairs(trait, pairs)
                total_added += added
                print(f"got {len(pairs)} raw, added {added} unique")
            except Exception as e:
                print(f"ERROR: {e} — retrying ...")
                continue

        final_count = len(load_existing(trait))
        print(f"  {trait}: done. Total pairs: {final_count}")

    print("\nGeneration complete. Run experiments/01_extract.py to re-extract.")


if __name__ == "__main__":
    main()
