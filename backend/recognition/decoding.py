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


def sentence_end_peak(boundary_probabilities: Sequence[Sequence[float]]) -> tuple[int, float]:
    """Find the strongest explicit sentence-end frame in a recent window."""
    if not boundary_probabilities:
        return -1, 0.0
    if any(len(frame) != 5 for frame in boundary_probabilities):
        raise ValueError("boundary probabilities must have five classes")
    peaks = [(index, float(frame[4])) for index, frame in enumerate(boundary_probabilities)]
    return max(peaks, key=lambda item: item[1])
