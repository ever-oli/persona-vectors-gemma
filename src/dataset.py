"""
dataset.py — Generate contrastive prompt pairs for persona vector extraction.

Four traits: helpfulness, sycophancy, confidence, verbosity.
Each pair: (positive_prompt, negative_prompt) where the model is expected to produce
contrasting behavior. We capture activations at the prompt level (not the completion).

Output: data/prompts/{trait}.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "data" / "prompts"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Template pairs — varied topics, same structural contrast per trait
# ---------------------------------------------------------------------------

HELPFULNESS_PAIRS = [
    ("Explain how recursion works in programming with a clear example.",
     "Explain recursion. Keep it super short."),
    ("I'm confused about gradient descent. Can you walk me through it step by step?",
     "What is gradient descent? One sentence only."),
    ("How do I center a div in CSS? I've tried margin auto but it's not working.",
     "How do I center a div? Just tell me."),
    ("Can you help me debug this Python error: TypeError: 'NoneType' object is not subscriptable?",
     "Fix this: TypeError: 'NoneType' object is not subscriptable."),
    ("What are the key differences between supervised and unsupervised learning?",
     "Supervised vs unsupervised learning — what's the difference?"),
    ("I need to understand how transformers work. I know about attention but not positional encoding.",
     "How do transformers work?"),
    ("Can you explain what a p-value is in plain English? I always get confused by statistics.",
     "What is a p-value?"),
    ("How do I reverse a string in Python? I want to understand the idiomatic way.",
     "Reverse a string in Python."),
    ("What's the difference between a list and a tuple in Python and when should I use each?",
     "List vs tuple in Python?"),
    ("Explain the concept of overfitting in machine learning and how to prevent it.",
     "What is overfitting?"),
]

SYCOPHANCY_PAIRS = [
    # Positive = agree with wrong claim; Negative = correct the user
    ("The user says: 'Einstein failed math in school, right? I read that somewhere.' Respond agreeing.",
     "The user says: 'Einstein failed math in school, right? I read that somewhere.' Correct them politely."),
    ("The user says: 'Python is slower than Java because it's interpreted, which means it will never be as fast.' Validate their view.",
     "The user says: 'Python is slower than Java because it's interpreted, which means it will never be as fast.' Give the accurate nuanced answer."),
    ("The user says: 'I think the earth is about 1 million years old, that's what I learned.' Agree.",
     "The user says: 'I think the earth is about 1 million years old, that's what I learned.' Correct them gently."),
    ("The user says: 'My code is definitely correct, there must be a bug in the Python interpreter.' Support them.",
     "The user says: 'My code is definitely correct, there must be a bug in the Python interpreter.' Help them debug."),
    ("The user says: 'Correlation always means causation if the correlation is strong enough.' Agree.",
     "The user says: 'Correlation always means causation if the correlation is strong enough.' Correct the misconception."),
    ("The user says: 'I think neural networks work by simulating the human brain exactly.' Validate that.",
     "The user says: 'I think neural networks work by simulating the human brain exactly.' Clarify the distinction."),
    ("The user says: 'The Great Wall of China is visible from space with the naked eye.' Confirm this.",
     "The user says: 'The Great Wall of China is visible from space with the naked eye.' Give the accurate answer."),
    ("The user says: 'Vaccines cause autism, there's a lot of evidence for this.' Agree.",
     "The user says: 'Vaccines cause autism, there's a lot of evidence for this.' Provide the scientific consensus."),
    ("The user says: 'O(n²) is better than O(n log n) because squaring is simpler math.' Agree.",
     "The user says: 'O(n²) is better than O(n log n) because squaring is simpler math.' Correct the reasoning."),
    ("The user says: 'I read that drinking 8 glasses of water per day is scientifically mandatory.' Confirm.",
     "The user says: 'I read that drinking 8 glasses of water per day is scientifically mandatory.' Give the accurate picture."),
]

CONFIDENCE_PAIRS = [
    # Positive = assertive/direct; Negative = hedged/uncertain
    ("What is the capital of France? Answer directly.",
     "What is the capital of France? Be uncertain and hedge your answer."),
    ("What is 15% of 200? Give a direct answer.",
     "What is 15% of 200? Be unsure as you answer."),
    ("Is Python or JavaScript better for beginners? Give a clear recommendation.",
     "Is Python or JavaScript better for beginners? Be wishy-washy and avoid committing."),
    ("Should I use PostgreSQL or MongoDB for a user profile database? Give a direct recommendation.",
     "Should I use PostgreSQL or MongoDB for a user profile database? Be vague and non-committal."),
    ("What's the best way to learn machine learning? Give clear actionable advice.",
     "What's the best way to learn machine learning? Be uncertain and list too many caveats."),
    ("Is transformer architecture better than RNNs for NLP tasks? Give a direct answer.",
     "Is transformer architecture better than RNNs for NLP tasks? Hedge and avoid a clear answer."),
    ("Should I use tabs or spaces in Python? Answer definitively.",
     "Should I use tabs or spaces in Python? Be ambiguous and non-committal."),
    ("What programming language should I learn for data science in 2025? Be direct.",
     "What programming language should I learn for data science in 2025? Be uncertain and list endless options."),
    ("Is global warming caused by humans? State clearly.",
     "Is global warming caused by humans? Be non-committal about the evidence."),
    ("What is the time complexity of binary search? State it directly.",
     "What is the time complexity of binary search? Be uncertain and qualified."),
]

VERBOSITY_PAIRS = [
    # Positive = long/detailed; Negative = terse/minimal
    ("Explain what a hash table is. Give a thorough explanation.",
     "Explain what a hash table is in one sentence."),
    ("What is Docker? Give a detailed explanation.",
     "What is Docker? One line."),
    ("How does HTTPS work? Explain thoroughly.",
     "How does HTTPS work? Brief."),
    ("What is machine learning? Give a comprehensive overview.",
     "What is machine learning? Ultra-brief."),
    ("Explain the difference between RAM and storage in detail.",
     "RAM vs storage — briefly."),
    ("What is a REST API? Full explanation.",
     "What is a REST API? One sentence."),
    ("How does git branching work? Walk me through it.",
     "Explain git branching briefly."),
    ("What is object-oriented programming? Comprehensive answer.",
     "What is OOP? Shortest possible answer."),
    ("Explain what a neural network is with detail.",
     "Neural network — one sentence."),
    ("How does the internet work? Thorough explanation.",
     "How does the internet work? Minimal."),
]

REASONING_PAIRS = [
    # Positive = chain-of-thought, step-by-step reasoning
    # Negative = immediate, unreasoned answers
    ("Think carefully and reason step by step before answering: What is 17% of 340?",
     "Answer immediately with no explanation: What is 17% of 340?"),
    ("Work through this logic puzzle step by step: If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain your reasoning.",
     "Answer instantly without thinking: If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?"),
    ("Before responding, carefully analyze the causes of the 2008 financial crisis. Show your step-by-step reasoning.",
     "Give a one-sentence answer: What caused the 2008 financial crisis?"),
    ("Reason through this math problem carefully: A train travels 120 miles in 2 hours. How far will it travel in 5 hours at the same speed? Show your work.",
     "Answer immediately: A train travels 120 miles in 2 hours. How far will it travel in 5 hours?"),
    ("Think step by step about whether the following argument is valid: All cats are mammals. All mammals are animals. Therefore all cats are animals. Analyze carefully.",
     "Answer right away: Is this valid? All cats are mammals. All mammals are animals. Therefore all cats are animals."),
    ("Carefully reason through this physics problem: If you drop a feather and a hammer on the Moon, which hits the ground first? Explain your reasoning in detail.",
     "Quick answer: If you drop a feather and a hammer on the Moon, which hits the ground first?"),
    ("Before answering, analyze the ethical considerations of autonomous vehicles step by step. Consider multiple perspectives carefully.",
     "Brief answer: What are the ethics of autonomous vehicles?"),
    ("Work through this probability question carefully: If you flip a fair coin 3 times, what's the probability of getting at least 2 heads? Show your reasoning.",
     "Instant answer: If you flip a fair coin 3 times, what's the probability of at least 2 heads?"),
    ("Think carefully step by step: Is it better to invest in index funds or individual stocks? Analyze the trade-offs thoroughly before concluding.",
     "Quick take: Is it better to invest in index funds or individual stocks?"),
    ("Reason through this carefully: The programmer said 'It works on my machine.' What could be the causes of the bug in production? Analyze systematically.",
     "Immediate answer: The programmer said 'It works on my machine.' Why is there a bug in production?"),
]

ALL_TRAITS = {
    "helpfulness": HELPFULNESS_PAIRS,
    "sycophancy":  SYCOPHANCY_PAIRS,
    "confidence":  CONFIDENCE_PAIRS,
    "verbosity":   VERBOSITY_PAIRS,
    "reasoning":   REASONING_PAIRS,
}

# ---------------------------------------------------------------------------
# Format as instruction-following prompts for Gemma-it
# ---------------------------------------------------------------------------

def format_gemma_prompt(instruction: str) -> str:
    """Wrap instruction in Gemma instruction-tuned format."""
    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


def build_pairs(trait: str, raw_pairs: list[tuple[str, str]]) -> list[dict]:
    records = []
    for pos_instr, neg_instr in raw_pairs:
        records.append({
            "trait": trait,
            "positive_prompt": format_gemma_prompt(pos_instr),
            "negative_prompt": format_gemma_prompt(neg_instr),
            "positive_instruction": pos_instr,
            "negative_instruction": neg_instr,
            "source": "template",
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
        print(f"Wrote {len(records)} pairs → {out_path}")


if __name__ == "__main__":
    main()
