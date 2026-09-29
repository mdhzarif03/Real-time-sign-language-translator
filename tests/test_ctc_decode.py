from __future__ import annotations

import unittest

from backend.recognition.decoding import ctc_greedy_decode, sentence_end_peak


class CtcDecoderTests(unittest.TestCase):
    def test_collapses_adjacent_frames_but_preserves_sign_repeated_after_blank(self) -> None:
        self.assertEqual(ctc_greedy_decode([0, 2, 2, 0, 2, 3, 3, 0]), [2, 2, 3])

    def test_empty_or_blank_only_sequence_is_empty(self) -> None:
        self.assertEqual(ctc_greedy_decode([]), [])
        self.assertEqual(ctc_greedy_decode([0, 0]), [])

    def test_single_frame_sentence_end_peak_is_not_diluted_by_neighboring_frames(self) -> None:
        probabilities = [
            [0.97, 0.01, 0.01, 0.01, 0.00],
            [0.01, 0.01, 0.01, 0.96, 0.01],  # sign end, not sentence end
            [0.97, 0.01, 0.01, 0.01, 0.00],
            [0.10, 0.01, 0.01, 0.01, 0.87],  # one-frame sentence boundary
        ]
        self.assertEqual(sentence_end_peak(probabilities), (3, 0.87))
        self.assertEqual(sentence_end_peak([]), (-1, 0.0))
        with self.assertRaisesRegex(ValueError, "five classes"):
            sentence_end_peak([[0.0, 1.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
