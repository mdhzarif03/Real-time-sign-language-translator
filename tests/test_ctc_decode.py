from __future__ import annotations

import unittest

from backend.recognition.decoding import ctc_greedy_decode


class CtcDecoderTests(unittest.TestCase):
    def test_collapses_adjacent_frames_but_preserves_sign_repeated_after_blank(self) -> None:
        self.assertEqual(ctc_greedy_decode([0, 2, 2, 0, 2, 3, 3, 0]), [2, 2, 3])

    def test_empty_or_blank_only_sequence_is_empty(self) -> None:
        self.assertEqual(ctc_greedy_decode([]), [])
        self.assertEqual(ctc_greedy_decode([0, 0]), [])


if __name__ == "__main__":
    unittest.main()
