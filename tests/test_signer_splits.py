from __future__ import annotations

import unittest

from training.split_by_signer import split_rows


class SignerSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"sample_id": f"{signer}-clip-{index}", "signer_id": signer}
            for signer in ("signer-a", "signer-b", "signer-c", "signer-d", "signer-e")
            for index in range(2)
        ]

    def test_split_is_deterministic_and_signer_disjoint(self) -> None:
        first = split_rows(self.rows, seed=42, train_ratio=0.6, validation_ratio=0.2)
        second = split_rows(self.rows, seed=42, train_ratio=0.6, validation_ratio=0.2)
        self.assertEqual(first, second)
        assignments: dict[str, set[str]] = {}
        for row in first:
            assignments.setdefault(row["signer_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in assignments.values()))
        self.assertEqual({row["split"] for row in first}, {"train", "validation", "test"})

    def test_duplicate_sample_ids_are_rejected(self) -> None:
        rows = [*self.rows, self.rows[0]]
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            split_rows(rows, seed=1, train_ratio=0.6, validation_ratio=0.2)

    def test_existing_split_assignments_are_rejected(self) -> None:
        rows = [dict(self.rows[0], split="train"), *self.rows[1:]]
        with self.assertRaisesRegex(ValueError, "must not already contain split"):
            split_rows(rows, seed=1, train_ratio=0.6, validation_ratio=0.2)

    def test_insufficient_signers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            split_rows(self.rows[:4], seed=1, train_ratio=0.6, validation_ratio=0.2)


if __name__ == "__main__":
    unittest.main()
