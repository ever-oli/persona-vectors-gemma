"""
dataset_base.py — Same contrastive pairs, plain-text format for base model.

No chat template — base models haven't seen <start_of_turn> tokens.
Format:
  Instruction: {instruction}
  Response:

This tests whether Paper 12's layer-0 finding was a template artifact.
"""
from __future__ import annotations
import json
from pathlib import Path

# Import raw pairs from dataset.py
from src.dataset import (
    HELPFULNESS_PAIRS,
    SYCOPHANCY_PAIRS,
    CONFIDENCE_PAIRS,
    VERBOSITY_PAIRS,
    ALL_TRAITS,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prompts_base"
OUT.mkdir(parents=True, exist_ok=True)


def format_plain_prompt(instruction: str) -> str:
    """Plain text format — no chat template."""
    return f"Instruction: {instruction}\nResponse:"


def build_pairs(trait: str, raw_pairs: list[tuple[str, str]]) -> list[dict]:
    records = []
    for pos_instr, neg_instr in raw_pairs:
        records.append({
            "trait": trait,
            "positive_prompt": format_plain_prompt(pos_instr),
            "negative_prompt": format_plain_prompt(neg_instr),
            "positive_instruction": pos_instr,
            "negative_instruction": neg_instr,
            "source": "template_plain",
        })
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    for trait, raw_pairs in ALL_TRAITS.items():
        records = build_pairs(trait, raw_pairs)
        out_path = OUT / f"{trait}.jsonl"
        write_jsonl(out_path, records)
        print(f"Wrote {len(records)} plain-text pairs → {out_path}")


if __name__ == "__main__":
    main()
