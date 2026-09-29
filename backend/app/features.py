"""Stable landmark tensor layout shared by preprocessing and inference."""

from __future__ import annotations

import numpy as np

from .schemas import Observation

HAND_POINTS = 21
POSE_POINTS = 33
FACE_POINTS = 478
CHANNELS = 4  # normalized x, y, z and visibility
FEATURE_DIM = (HAND_POINTS * 2 + POSE_POINTS + FACE_POINTS) * CHANNELS


def _append_points(target: np.ndarray, offset: int, points: list | None, count: int) -> None:
    if not points:
        return
    for index, point in enumerate(points[:count]):
        base = offset + index * CHANNELS
        target[base : base + CHANNELS] = (point.x, point.y, point.z, point.visibility)


def observation_to_features(observation: Observation) -> np.ndarray:
    """Map one observation into a fixed 2,212-float layout; absent landmarks are zeroed."""
    result = np.zeros(FEATURE_DIM, dtype=np.float32)
    hand_size = HAND_POINTS * CHANNELS
    pose_size = POSE_POINTS * CHANNELS
    face_size = FACE_POINTS * CHANNELS
    _append_points(result, 0, observation.landmarks.left_hand, HAND_POINTS)
    _append_points(result, hand_size, observation.landmarks.right_hand, HAND_POINTS)
    _append_points(result, hand_size * 2, observation.landmarks.pose, POSE_POINTS)
    _append_points(result, hand_size * 2 + pose_size, observation.landmarks.face, FACE_POINTS)
    return result
