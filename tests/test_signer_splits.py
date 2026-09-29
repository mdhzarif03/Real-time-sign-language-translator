from __future__ import annotations

import unittest

from training.split_by_signer import split_rows


class SignerSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"sample_id": f"{signer}-clip-{index}", "signer_id": signer, "session_id": f"{signer}-session-{index}"}
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

    def test_signers_in_shared_sessions_stay_in_the_same_split(self) -> None:
        rows = [
            *self.rows,
            {"sample_id": "shared-session-extra", "signer_id": "signer-f", "session_id": "signer-a-session-0"},
        ]
        split = split_rows(rows, seed=42, train_ratio=0.6, validation_ratio=0.2)
        assignments: dict[str, set[str]] = {}
        for row in split:
            assignments.setdefault(row["session_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in assignments.values()))
        self.assertEqual(split[0]["split"], next(row["split"] for row in split if row["signer_id"] == "signer-f"))

    def test_missing_session_identity_is_rejected(self) -> None:
        rows = [dict(self.rows[0], session_id=""), *self.rows[1:]]
        with self.assertRaisesRegex(ValueError, "session_id"):
            split_rows(rows, seed=1, train_ratio=0.6, validation_ratio=0.2)


if __name__ == "__main__":
    unittest.main()
