from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from backend.app.features import FEATURE_DIM
    from backend.recognition.runtime import FEATURE_LAYOUT, TemporalRuntime
    from backend.recognition.temporal_model import build_model_from_config


@unittest.skipUnless(torch is not None, "install training/requirements.txt to run checkpoint runtime tests")
class TemporalRuntimeTests(unittest.TestCase):
    def test_checkpoint_load_language_gate_and_timestamped_inference(self) -> None:
        config = {
            "feature_dim": FEATURE_DIM,
            "vocabulary_size": 2,
            "architecture": "spatiotemporal-landmark-ctc-v3",
            "width": 64,
            "heads": 4,
            "layers": 1,
            "feedforward_dim": 128,
            "dropout": 0,
        }
        model = build_model_from_config(config, vocabulary_size=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "model.pt"
            torch.save({
                "model_state": model.state_dict(),
                "model_config": config,
                "sign_language": "en-US-ASL",
                "vocabulary": {"HELLO": 1, "WORLD": 2},
                "feature_layout": FEATURE_LAYOUT,
            }, checkpoint_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "model_id": "test-model",
                "model_version": "test-1",
                "sign_language": "en-US-ASL",
                "checkpoint": "model.pt",
                "feature_layout": FEATURE_LAYOUT,
                "window_size": 16,
            }), encoding="utf-8")
            runtime, status = TemporalRuntime.load(manifest_path)

        self.assertIsNotNone(runtime)
        self.assertEqual(status.state, "ready")
        self.assertEqual(runtime.status_for("bn-BD-BdSL").state, "model_unavailable")
        predictions = runtime.predict([(np.zeros(FEATURE_DIM, dtype=np.float32), index * 80) for index in range(8)], "test", 0)
        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.model_version, "test-1")
        self.assertGreaterEqual(predictions.latency_ms, 0)
        self.assertTrue(all(token.end_ms >= token.start_ms for token in predictions.tokens))


if __name__ == "__main__":
    unittest.main()
