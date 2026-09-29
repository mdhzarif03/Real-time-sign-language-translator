"""Train the landmark temporal CTC baseline on a signer-split JSONL manifest."""

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

from backend.app.features import FEATURE_DIM
from backend.recognition.temporal_model import TemporalSignTransformer, ctc_objective


class SequenceDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, vocabulary_size: int) -> None:
        self.rows = rows
        self.root = root
        self.vocabulary_size = vocabulary_size

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
        if boundaries.shape != (features.shape[0],) or np.any((boundaries < 0) | (boundaries > 3)):
            raise ValueError(f"{row['sample_id']}: boundary labels must be [time] integers in 0..3")
        if np.any((gloss_ids < 1) | (gloss_ids >= self.vocabulary_size)):
            raise ValueError(f"{row['sample_id']}: gloss ID outside the language vocabulary")
        return torch.from_numpy(features), torch.from_numpy(gloss_ids), torch.from_numpy(boundaries)

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
    for row in rows:
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("each row needs train, validation or test split metadata")
        sample_id = row.get("sample_id")
        signer_id = row.get("signer_id")
        if not sample_id or sample_id in seen or not signer_id:
            raise ValueError("manifest sample IDs must be unique and signer IDs must be present")
        if row.get("language") != language:
            raise ValueError(f"{sample_id}: sample language must match the requested model language {language}")
        seen.add(sample_id)
        signer_splits.setdefault(signer_id, set()).add(row["split"])
    leaked = [signer for signer, splits in signer_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"signer leakage across splits: {', '.join(leaked[:10])}")
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
    previous = None
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
    parser.add_argument("vocabulary", type=Path, help="JSON object mapping gloss strings to IDs starting at 1")
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", required=True, help="explicit sign-language identifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows, signer_splits = load_manifest(args.manifest, args.language)
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    if not isinstance(vocabulary, dict) or not vocabulary or sorted(vocabulary.values()) != list(range(1, len(vocabulary) + 1)):
        raise ValueError("vocabulary IDs must be a contiguous integer range beginning at 1; CTC blank is reserved at 0")
    root = args.manifest.parent
    datasets = {name: SequenceDataset([row for row in rows if row["split"] == name], root, len(vocabulary) + 1) for name in ("train", "validation", "test")}
    loaders = {name: DataLoader(data, batch_size=args.batch_size, shuffle=name == "train", collate_fn=collate, num_workers=0) for name, data in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalSignTransformer(FEATURE_DIM, len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    best_wer = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0
        for features, targets, lengths, target_lengths, boundary_targets in loaders["train"]:
            features = features.to(device); targets = targets.to(device); lengths = lengths.to(device)
            output = model(features, lengths)
            loss = ctc_objective(output, targets, target_lengths.to(device), lengths, boundary_targets.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += float(loss.detach())
        val_wer = validation_wer(model, loaders["validation"], device)
        print(f"epoch={epoch} train_loss={total_loss / max(len(loaders['train']), 1):.4f} validation_gloss_wer={val_wer:.4f}")
        if val_wer < best_wer:
            best_wer = val_wer
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {
                    "feature_dim": FEATURE_DIM,
                    "vocabulary_size": len(vocabulary),
                    "architecture": "spatiotemporal-landmark-ctc-v2",
                    "width": model.encoder.layers[0].self_attn.embed_dim,
                    "heads": model.encoder.layers[0].self_attn.num_heads,
                    "layers": len(model.encoder.layers),
                    "feedforward_dim": model.encoder.layers[0].linear1.out_features,
                    "dropout": model.encoder.layers[0].dropout.p,
                },
                "sign_language": args.language,
                "model_version": "signflow-spatiotemporal-ctc-v2",
                "vocabulary": vocabulary,
                "feature_layout": "hands-left-right-21x4_pose-33x4_face-478x4_v1",
                "dataset_manifest": str(args.manifest),
                "dataset_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "vocabulary_sha256": hashlib.sha256(args.vocabulary.read_bytes()).hexdigest(),
                "split_signer_counts": {split: sum(split in labels for labels in signer_splits.values()) for split in ("train", "validation", "test")},
                "epoch": epoch,
                "validation_gloss_wer": val_wer,
                "training_device": str(device),
                "torch_version": str(torch.__version__),
            }, args.output)
    print(f"best_validation_gloss_wer={best_wer:.4f}; checkpoint={args.output}")
    best = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    test_wer = validation_wer(model, loaders["test"], device)
    print(f"held_out_test_gloss_wer={test_wer:.4f} (best validation checkpoint, evaluated once)")


if __name__ == "__main__":
    main()
