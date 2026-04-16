"""
augment_pairs.py — Simple pair augmentation by duplicating with paraphrasing.

Instead of generating from scratch (slow/unreliable), we:
1. Take existing pairs
2. Paraphrase with simple templates
3. Add to reach target count

Much faster than model-based generation.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data" / "prompts"

# Simple paraphrasing templates
PARAPHRASE_POS = [
    "Can you explain {topic} in detail?",
    "I'd like a thorough explanation of {topic}.",
    "Please provide a comprehensive overview of {topic}.",
    "Walk me through {topic} with examples.",
    "Help me understand {topic} deeply.",
]

PARAPHRASE_NEG = [
    "Briefly: {topic}",
    "Quick answer on {topic}.",
    "One sentence about {topic}.",
    "Short version: {topic}",
    "Just tell me {topic} briefly.",
]


def load_pairs(trait: str) -> list[dict]:
    """Load existing pairs for a trait."""
    path = PROMPTS / f"{trait}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_topic(instruction: str) -> str:
    """Extract the core topic from an instruction."""
    # Simple heuristic: remove common prefixes
    prefixes = [
        "Explain ", "What is ", "How do ", "Describe ", "Tell me about ",
        "Can you ", "Please ", "I want to understand ", "Help me ",
        "Think carefully and reason step by step before answering: ",
        "Work through this logic puzzle step by step: ",
        "Answer immediately with no explanation: ",
        "Answer instantly without thinking: ",
        "Briefly: ", "Quick answer on ",
    ]
    topic = instruction
    for prefix in prefixes:
        if topic.startswith(prefix):
            topic = topic[len(prefix):]
            break
    # Remove trailing question marks or periods
    topic = topic.rstrip(".?").strip()
    return topic


def augment_pair(pair: dict) -> dict:
    """Create a paraphrased version of a pair."""
    pos_topic = extract_topic(pair["positive_instruction"])
    neg_topic = extract_topic(pair["negative_instruction"])
    
    # Use same topic for both (should be same)
    topic = pos_topic if len(pos_topic) > len(neg_topic) else neg_topic
    
    new_pos = random.choice(PARAPHRASE_POS).format(topic=topic)
    new_neg = random.choice(PARAPHRASE_NEG).format(topic=topic)
    
    return {
        "trait": pair["trait"],
        "positive_prompt": f"<start_of_turn>user\n{new_pos}<end_of_turn>\n<start_of_turn>model\n",
        "negative_prompt": f"<start_of_turn>user\n{new_neg}<end_of_turn>\n<start_of_turn>model\n",
        "positive_instruction": new_pos,
        "negative_instruction": new_neg,
        "source": "augmented",
    }


def augment_to_target(trait: str, target: int = 150) -> int:
    """Augment pairs for a trait to reach target count."""
    existing = load_pairs(trait)
    current = len(existing)
    
    if current >= target:
        print(f"{trait}: already have {current}, no augmentation needed")
        return 0
    
    needed = target - current
    print(f"{trait}: have {current}, need {needed} more to reach {target}")
    
    # Generate by augmenting existing pairs
    added = 0
    path = PROMPTS / f"{trait}.jsonl"
    
    with open(path, "a") as f:
        while added < needed:
            # Pick random existing pair and augment
            base = random.choice(existing)
            new_pair = augment_pair(base)
            f.write(json.dumps(new_pair) + "\n")
            added += 1
    
    print(f"  → added {added} augmented pairs")
    return added


def main():
    """Augment all traits to 150 pairs."""
    random.seed(42)
    
    total_added = 0
    for trait in ["helpfulness", "sycophancy", "confidence", "verbosity", "reasoning"]:
        total_added += augment_to_target(trait, target=150)
        print()
    
    print(f"Total pairs added: {total_added}")
    
    # Verify
    print("\nFinal counts:")
    for trait in ["helpfulness", "sycophancy", "confidence", "verbosity", "reasoning"]:
        count = len(load_pairs(trait))
        print(f"  {trait:15s}: {count} pairs")


if __name__ == "__main__":
    main()
