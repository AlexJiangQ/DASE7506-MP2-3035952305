"""One fresh-process synthetic prediction + scoring timing worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import parse_answer
import submission


def make_bank(n: int) -> list[dict]:
    templates = [
        "1 + 1 = 2. Final Answer: \\boxed{2}",
        "We compute carefully. #### 2",
        "1 + 1 = 3. Final Answer: \\boxed{3}",
        "No numeric answer is available.",
    ]
    return [{
        "id": f"synthetic-audit::{i * 101 + 9}",
        "question": f"What is 1 + 1? Wording variant {i}",
        "candidates": [
            {"index": j, "text": text + (" detail" * (i % 4))}
            for j, text in enumerate(templates)
        ],
    } for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=798)
    args = parser.parse_args()
    bank = make_bank(args.size)
    rng = np.random.default_rng(20261022)
    embeddings = rng.normal(size=(args.size, 4, 1024)).astype(np.float32)
    prediction_started = time.perf_counter()
    choices = submission.predict(bank, {"embeddings": embeddings}, args.checkpoint)
    prediction_seconds = time.perf_counter() - prediction_started
    scoring_started = time.perf_counter()
    gold = "2"
    correct = sum(
        parse_answer(row["candidates"][choice]["text"]) == gold
        for row, choice in zip(bank, choices)
    )
    scoring_seconds = time.perf_counter() - scoring_started
    output = {
        "kind": "synthetic_audit",
        "n": args.size,
        "selected": choices,
        "correct": correct,
        "accuracy": 100.0 * correct / args.size,
        "prediction_seconds": prediction_seconds,
        "scoring_seconds": scoring_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "selected"}))


if __name__ == "__main__":
    main()
