from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from backend.app.features import FEATURE_DIM
    from backend.recognition.temporal_model import TemporalSignTransformer, build_model_from_config, ctc_objective


@unittest.skipUnless(torch is not None, "install training/requirements.txt to run model tensor tests")
class TemporalModelTests(unittest.TestCase):
    def test_checkpoint_configuration_is_bounded_and_recreates_architecture(self) -> None:
        config = {
            "architecture": "spatiotemporal-landmark-ctc-v4-gru",
            "feature_dim": FEATURE_DIM,
            "vocabulary_size": 7,
            "width": 64,
            "heads": 4,
            "layers": 2,
            "feedforward_dim": 128,
            "dropout": 0.1,
        }
        model = build_model_from_config(config, vocabulary_size=7)
        self.assertEqual(model.temporal.hidden_size, 64)
        self.assertEqual(model.temporal.num_layers, 2)
        with self.assertRaises(ValueError):
            build_model_from_config({**config, "width": 16_384}, vocabulary_size=7)

    def test_spatial_temporal_heads_have_finite_logits_and_gradients(self) -> None:
        model = TemporalSignTransformer(
            FEATURE_DIM,
            vocabulary_size=5,
            width=64,
            heads=4,
            layers=1,
            feedforward_dim=128,
            dropout=0,
            max_length=16,
        )
        features = torch.zeros(2, 8, FEATURE_DIM)
        features[0, :, 3] = 1  # left-hand visibility
        features[1, :, 42 * 4 + 3] = 1  # one pose-joint visibility
        lengths = torch.tensor([8, 6])
        output = model(features, lengths)

        self.assertEqual(output.sign_logits.shape, (2, 8, 6))
        self.assertEqual(output.boundary_logits.shape, (2, 8, 5))
        self.assertTrue(torch.isfinite(output.sign_logits).all())
        boundaries = torch.zeros(2, 8, dtype=torch.long)
        boundaries[1, 6:] = -100
        loss = ctc_objective(
            output,
            torch.tensor([1, 2, 3]),
            torch.tensor([2, 1]),
            lengths,
            boundaries,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.hand_encoder.input_projection.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
