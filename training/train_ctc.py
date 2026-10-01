"""Train the low-latency landmark CTC recognizer on signer-disjoint data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.app.features import FEATURE_DIM, normalize_features
from backend.recognition.compute import configure_torch_threads
from backend.recognition.temporal_model import TemporalSignTransformer, ctc_objective


class SequenceDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, vocabulary_size: int, augment: bool = False, seed: int = 2026) -> None:
        self.rows = rows
        self.root = root
        self.vocabulary_size = vocabulary_size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        features = np.load(self._safe_path(row["feature_file"]), allow_pickle=False).astype(np.float32, copy=False)
        boundaries = np.load(self._safe_path(row["boundary_file"]), allow_pickle=False).astype(np.int64, copy=False)
        gloss_ids = np.asarray(row["gloss_ids"], dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM or features.shape[0] < 1:
            raise ValueError(f"{row['sample_id']}: expected non-empty [time,{FEATURE_DIM}] features")
        if gloss_ids.ndim != 1 or gloss_ids.size == 0:
            raise ValueError(f"{row['sample_id']}: gloss_ids must be a non-empty sequence")
        if boundaries.shape != (features.shape[0],) or np.any((boundaries < 0) | (boundaries > 4)):
            raise ValueError(f"{row['sample_id']}: boundary labels must be [time] integers in 0..4")
        if np.count_nonzero(boundaries == 4) != 1 or np.count_nonzero(boundaries == 1) == 0:
            raise ValueError(f"{row['sample_id']}: each sentence needs sign onsets and exactly one sentence-end boundary")
        if np.any((gloss_ids < 1) | (gloss_ids >= self.vocabulary_size)):
            raise ValueError(f"{row['sample_id']}: gloss ID outside the language vocabulary")
        features = normalize_features(features)
        if self.augment:
            features = self._augment(features, index)
        return torch.from_numpy(features), torch.from_numpy(gloss_ids), torch.from_numpy(boundaries)

    def _augment(self, features: np.ndarray, index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + index * 1_000_003)
        output = features.copy()
        # Landmark jitter is intentionally small because the inputs are already normalized.
        output[:, :,] += rng.normal(0.0, 0.008, output.shape).astype(np.float32)
        output[..., 3] = np.clip(output[..., 3], 0.0, 1.0)
        # Randomly hide an entire anatomical stream occasionally to improve robustness to detector misses.
        if rng.random() < 0.18:
            group = int(rng.integers(0, 4))
            starts = (0, 21 * 4, 42 * 4, 75 * 4)
            counts = (21 * 4, 21 * 4, 33 * 4, 478 * 4)
            output[:, starts[group] : starts[group] + counts[group]] = 0
        # Mild temporal jitter by dropping/repeating a few internal frames.
        if len(output) >= 20 and rng.random() < 0.20:
            keep = np.arange(len(output))
            if len(keep) > 24:
                drop_count = max(1, len(keep) // 20)
                drop = rng.choice(np.arange(2, len(keep) - 2), size=drop_count, replace=False)
                keep = np.delete(keep, np.sort(drop))
                output = output[keep]
        return output.astype(np.float32, copy=False)

    def _safe_path(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("feature and boundary paths must be non-empty relative paths")
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("manifest feature paths must remain within the manifest directory")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def collate(batch):
    sequences, targets, boundaries = zip(*batch, strict=True)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    padded_boundaries = torch.full(padded_sequences.shape[:2], -100, dtype=torch.long)
    for index, labels in enumerate(boundaries):
        padded_boundaries[index, : len(labels)] = labels
    return padded_sequences, torch.cat(targets), lengths, target_lengths, padded_boundaries


def load_manifest(path: Path, language: str) -> tuple[list[dict], dict[str, set[str]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    signer_splits: dict[str, set[str]] = {}
    session_splits: dict[str, set[str]] = {}
    for row in rows:
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("each row needs train, validation or test split metadata")
        sample_id = row.get("sample_id")
        signer_id = row.get("signer_id")
        session_id = row.get("session_id")
        if not all(isinstance(value, str) and value.strip() for value in (sample_id, signer_id, session_id)) or sample_id in seen:
            raise ValueError("manifest sample IDs must be unique and signer_id/session_id must be present")
        if row.get("language") != language:
            raise ValueError(f"{sample_id}: sample language must match the requested model language {language}")
        seen.add(sample_id)
        signer_splits.setdefault(signer_id, set()).add(row["split"])
        session_splits.setdefault(session_id, set()).add(row["split"])
    leaked = [signer for signer, splits in signer_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"signer leakage across splits: {', '.join(leaked[:10])}")
    leaked_sessions = [session for session, splits in session_splits.items() if len(splits) > 1]
    if leaked_sessions:
        raise ValueError(f"recording session leakage across splits: {', '.join(leaked_sessions[:10])}")
    if not {"train", "validation", "test"}.issubset({row["split"] for row in rows}):
        raise ValueError("manifest must have samples in train, validation and test")
    return rows, signer_splits


def edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i]
        for j, hyp in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def decode(logits: torch.Tensor, length: int) -> list[int]:
    labels = logits[:length].argmax(dim=-1).tolist()
    result: list[int] = []
    previous = 0
    for label in labels:
        if label != 0 and label != previous:
            result.append(label)
        previous = label
    return result


@torch.no_grad()
def validation_wer(model, loader, device) -> float:
    model.eval()
    errors = words = 0
    for features, targets, lengths, target_lengths, _ in loader:
        output = model(features.to(device), lengths.to(device))
        target_offset = 0
        for index, (input_length, target_length) in enumerate(zip(lengths.tolist(), target_lengths.tolist(), strict=True)):
            reference = targets[target_offset : target_offset + target_length].tolist()
            hypothesis = decode(output.sign_logits[index], input_length)
            errors += edit_distance(reference, hypothesis)
            words += len(reference)
            target_offset += target_length
    return errors / max(words, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("vocabulary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        parser.error("epochs, batch-size, patience and lr must be positive")
    configure_torch_threads()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows, signer_splits = load_manifest(args.manifest, args.language)
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    if not isinstance(vocabulary, dict) or not vocabulary or sorted(vocabulary.values()) != list(range(1, len(vocabulary) + 1)):
        raise ValueError("vocabulary IDs must be a contiguous integer range beginning at 1; CTC blank is reserved at 0")
    root = args.manifest.parent
    datasets = {
        "train": SequenceDataset([r for r in rows if r["split"] == "train"], root, len(vocabulary) + 1, augment=True, seed=args.seed),
        "validation": SequenceDataset([r for r in rows if r["split"] == "validation"], root, len(vocabulary) + 1),
        "test": SequenceDataset([r for r in rows if r["split"] == "test"], root, len(vocabulary) + 1),
    }
    loaders = {
        name: DataLoader(data, batch_size=args.batch_size, shuffle=name == "train", collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())
        for name, data in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalSignTransformer(FEATURE_DIM, len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=2e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_wer = float("inf")
    stale = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0
        for features, targets, lengths, target_lengths, boundary_targets in loaders["train"]:
            features = features.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True); lengths = lengths.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(features, lengths)
                loss = ctc_objective(output, targets, target_lengths.to(device), lengths, boundary_targets.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += float(loss.detach())
        val_wer = validation_wer(model, loaders["validation"], device)
        scheduler.step(val_wer)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} train_loss={total_loss / max(len(loaders['train']), 1):.4f} validation_gloss_wer={val_wer:.4f} lr={lr:.2e}")
        if val_wer < best_wer - 1e-4:
            best_wer = val_wer; stale = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {
                    "feature_dim": FEATURE_DIM,
                    "vocabulary_size": len(vocabulary),
                    "architecture": "spatiotemporal-landmark-ctc-v4-gru",
                    "width": model.temporal.hidden_size,
                    "heads": 4,
                    "layers": model.temporal.num_layers,
                    "feedforward_dim": model.temporal.hidden_size * 2,
                    "dropout": model.temporal.dropout,
                },
                "sign_language": args.language,
                "model_version": "signflow-spatiotemporal-ctc-v4-gru",
                "vocabulary": vocabulary,
                "feature_layout": "hands-left-right-21x4_pose-33x4_face-478x4_v2-normalized",
                "dataset_manifest": str(args.manifest),
                "dataset_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "vocabulary_sha256": hashlib.sha256(args.vocabulary.read_bytes()).hexdigest(),
                "split_signer_counts": {split: sum(split in labels for labels in signer_splits.values()) for split in ("train", "validation", "test")},
                "epoch": epoch,
                "validation_gloss_wer": val_wer,
                "training_device": str(device),
                "torch_version": str(torch.__version__),
            }, args.output)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch} best_validation_gloss_wer={best_wer:.4f}")
                break
    print(f"best_validation_gloss_wer={best_wer:.4f}; checkpoint={args.output}")
    best = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    test_wer = validation_wer(model, loaders["test"], device)
    print(f"held_out_test_gloss_wer={test_wer:.4f} (best validation checkpoint, evaluated once)")


if __name__ == "__main__":
    main()
