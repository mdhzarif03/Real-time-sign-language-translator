"""Create a deterministic signer-disjoint split manifest from metadata JSONL."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split_rows(rows: list[dict], seed: int, train_ratio: float, validation_ratio: float) -> list[dict]:
    signer_samples: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        signer_id = row.get("signer_id")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(signer_id, str) or not signer_id:
            raise ValueError("every row must include non-empty sample_id and signer_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if "split" in row:
            raise ValueError("input manifest must not already contain split assignments")
        signer_samples[signer_id].append(row)

    signer_ids = sorted(signer_samples)
    if len(signer_ids) < 3:
        raise ValueError("at least three distinct signers are required for train/validation/test splits")
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("train and validation ratios must be positive and leave a test split")

    random.Random(seed).shuffle(signer_ids)
    count = len(signer_ids)
    validation_count = max(1, round(count * validation_ratio))
    test_count = max(1, round(count * (1 - train_ratio - validation_ratio)))
    train_count = count - validation_count - test_count
    if train_count < 1:
        raise ValueError("split ratios leave fewer than one signer in the train split")
    train_end = train_count
    validation_end = train_count + validation_count
    signer_split = {
        **{signer: "train" for signer in signer_ids[:train_end]},
        **{signer: "validation" for signer in signer_ids[train_end:validation_end]},
        **{signer: "test" for signer in signer_ids[validation_end:]},
    }
    return [{**row, "split": signer_split[row["signer_id"]]} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL manifest without split fields")
    parser.add_argument("output", type=Path, help="destination JSONL manifest")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--validation", type=float, default=0.1)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    split = split_rows(rows, args.seed, args.train, args.validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split), encoding="utf-8")
    signers: dict[str, set[str]] = defaultdict(set)
    for row in split:
        signers[row["split"]].add(row["signer_id"])
    print("signers:", {name: len(signers[name]) for name in ("train", "validation", "test")})
    print("samples:", {name: sum(row["split"] == name for row in split) for name in ("train", "validation", "test")})


if __name__ == "__main__":
    main()
