from __future__ import annotations

import unittest

from training.prepare_landmarks import _boundary_targets


class AnnotationPreprocessingTests(unittest.TestCase):
    def test_segments_map_to_boundary_labels_and_gloss_ids(self) -> None:
        glosses, boundaries = _boundary_targets(
            [0, 80, 160, 240, 320, 400],
            [
                {"gloss": "GO", "start_ms": 80, "end_ms": 240},
                {"gloss": "HOME", "start_ms": 320, "end_ms": 400},
            ],
            {"GO": 1, "HOME": 2},
        )
        self.assertEqual(glosses, [1, 2])
        self.assertEqual(boundaries, [0, 1, 2, 3, 1, 3])

    def test_missing_vocabulary_entry_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing from the supplied language vocabulary"):
            _boundary_targets([0, 80, 160], [{"gloss": "GO", "start_ms": 0, "end_ms": 160}], {})

    def test_segment_must_cover_multiple_sampled_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            _boundary_targets([0, 80, 160], [{"gloss": "GO", "start_ms": 80, "end_ms": 81}], {"GO": 1})


if __name__ == "__main__":
    unittest.main()
