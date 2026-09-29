from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from backend.app.features import FEATURE_DIM, landmarks_to_features, observation_to_features
from backend.app.schemas import Observation


def observation_with_hand() -> Observation:
    point = {"x": 0.25, "y": 0.5, "z": -0.1, "visibility": 0.8}
    return Observation.model_validate({
        "schema_version": 1,
        "type": "observation",
        "stream_id": "unit-test",
        "sequence": 0,
        "timestamp_ms": 1,
        "sign_language": "en-US-ASL",
        "image_width": 640,
        "image_height": 360,
        "landmarks": {
            "left_hand": [point] * 21,
            "right_hand": None,
            "pose": None,
            "face": None,
        },
    })


class FeatureLayoutTests(unittest.TestCase):
    def test_missing_landmarks_zero_fill_fixed_layout(self) -> None:
        features = observation_to_features(observation_with_hand())
        self.assertEqual(features.shape, (FEATURE_DIM,))
        self.assertEqual(FEATURE_DIM, 2212)
        self.assertTrue(np.all(features[84:] == 0))
        np.testing.assert_allclose(features[:4], [0.25, 0.5, -0.1, 0.8])

    def test_feature_tensor_is_float32(self) -> None:
        self.assertEqual(observation_to_features(observation_with_hand()).dtype, np.float32)

    def test_extractor_accepts_media_pipe_style_dictionaries(self) -> None:
        features = landmarks_to_features([{"x": 0.1, "y": 0.2, "z": 0.3, "visibility": 1}], None, None, None)
        np.testing.assert_allclose(features[:4], [0.1, 0.2, 0.3, 1])

    def test_landmark_values_outside_contract_are_rejected(self) -> None:
        payload = observation_with_hand().model_dump()
        payload["landmarks"]["left_hand"][0]["x"] = 9
        with self.assertRaises(ValidationError):
            Observation.model_validate(payload)

    def test_unknown_fields_are_rejected(self) -> None:
        payload = observation_with_hand().model_dump()
        payload["unexpected"] = "raw video must not be accepted"
        with self.assertRaises(ValidationError):
            Observation.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
