from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

if torch is not None:
    from training.train_ctc import load_manifest


@unittest.skipUnless(torch is not None, "install training/requirements.txt to run training manifest tests")
class TrainingManifestTests(unittest.TestCase):
    def test_manifest_rejects_shared_recording_session_across_splits(self) -> None:
        rows = [
            {"sample_id": "train-1", "signer_id": "signer-1", "session_id": "shared-session", "split": "train", "language": "ASL"},
            {"sample_id": "test-1", "signer_id": "signer-2", "session_id": "shared-session", "split": "test", "language": "ASL"},
            {"sample_id": "validation-1", "signer_id": "signer-3", "session_id": "validation-session", "split": "validation", "language": "ASL"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recording session leakage"):
                load_manifest(manifest, "ASL")


if __name__ == "__main__":
    unittest.main()
