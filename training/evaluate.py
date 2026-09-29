"""Evaluate a trained landmark CTC checkpoint on its held-out signer split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from backend.app.features import FEATURE_DIM
from backend.recognition.temporal_model import build_model_from_config
from training.train_ctc import SequenceDataset, collate, decode, load_manifest


def _edit_counts(reference: list[int], hypothesis: list[int]) -> tuple[int, int, int, int, int, int]:
    """Return aligned token TP/FP/FN and sequence substitutions/deletions/insertions."""
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        costs[row][0] = row
    for column in range(columns):
        costs[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1] + (reference[row - 1] != hypothesis[column - 1]),
            )
    tp = fp = fn = substitutions = deletions = insertions = 0
    row, column = len(reference), len(hypothesis)
    while row or column:
        if row and column and reference[row - 1] == hypothesis[column - 1] and costs[row][column] == costs[row - 1][column - 1]:
            tp += 1
            row -= 1
            column -= 1
        elif row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            fn += 1; fp += 1; substitutions += 1
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            fn += 1; deletions += 1
            row -= 1
        else:
            fp += 1; insertions += 1
            column -= 1
    return tp, fp, fn, substitutions, deletions, insertions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="feature manifest with signer-independent splits")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    rows, signer_splits = load_manifest(args.manifest, args.language)
    held_out = [row for row in rows if row["split"] == "test"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocabulary_by_word = checkpoint.get("vocabulary")
    config = checkpoint.get("model_config", {})
    if checkpoint.get("sign_language") != args.language or checkpoint.get("feature_layout") != "hands-left-right-21x4_pose-33x4_face-478x4_v1":
        raise ValueError("checkpoint language or feature layout does not match the requested evaluation")
    if not isinstance(vocabulary_by_word, dict) or int(config.get("feature_dim", -1)) != FEATURE_DIM:
        raise ValueError("checkpoint is missing a compatible vocabulary/model configuration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config, len(vocabulary_by_word)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    dataset = SequenceDataset(held_out, args.manifest.parent, len(vocabulary_by_word) + 1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    substitutions = deletions = insertions = sequence_errors = reference_count = 0
    tp = fp = fn = 0
    latencies_ms: list[float] = []

    with torch.inference_mode():
        for features, targets, lengths, target_lengths, _ in loader:
            features = features.to(device)
            lengths_device = lengths.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(features, lengths_device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000 / len(features))
            offset = 0
            for index, (input_length, target_length) in enumerate(zip(lengths.tolist(), target_lengths.tolist(), strict=True)):
                reference = targets[offset : offset + target_length].tolist()
                hypothesis = decode(output.sign_logits[index], input_length)
                offset += target_length
                reference_count += len(reference)
                if reference != hypothesis:
                    sequence_errors += 1
                sample_tp, sample_fp, sample_fn, sample_subs, sample_dels, sample_ins = _edit_counts(reference, hypothesis)
                tp += sample_tp; fp += sample_fp; fn += sample_fn
                substitutions += sample_subs; deletions += sample_dels; insertions += sample_ins

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    report = {
        "dataset_manifest": str(args.manifest.resolve()),
        "dataset_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "language": args.language,
        "model_version": checkpoint.get("model_version", "unspecified"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "signer_independent_split": True,
        "test_signer_count": sum("test" in split_names for split_names in signer_splits.values()),
        "test_sample_count": len(held_out),
        "gloss_vocabulary_size": len(vocabulary_by_word),
        "metrics": {
            "gloss_wer": (substitutions + deletions + insertions) / max(reference_count, 1),
            "sign_error_rate": (substitutions + deletions + insertions) / max(reference_count, 1),
            "sentence_error_rate": sequence_errors / max(len(held_out), 1),
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
        },
        "model_forward_latency_ms_per_sequence": {
            "mean": float(np.mean(latencies_ms)) if latencies_ms else None,
            "p50": float(np.percentile(latencies_ms, 50)) if latencies_ms else None,
            "p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else None,
            "includes_preprocessing": False,
        },
        "runtime": {
            "device": str(device),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "checkpoint_reported_validation_wer": checkpoint.get("validation_gloss_wer"),
        },
        "interpretation": "Offline results on the supplied held-out signer split; not a real-world accuracy claim.",
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

