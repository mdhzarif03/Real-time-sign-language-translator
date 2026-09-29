from __future__ import annotations

from collections.abc import Sequence


def ctc_greedy_decode(labels: Sequence[int]) -> list[int]:
    """Collapse CTC repeats, allowing the same gloss again after a blank."""
    decoded: list[int] = []
    previous = 0
    for raw_label in labels:
        label = int(raw_label)
        if label != 0 and label != previous:
            decoded.append(label)
        previous = label
    return decoded
