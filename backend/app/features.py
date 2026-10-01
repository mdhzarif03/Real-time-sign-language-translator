"""Stable, normalized landmark tensor layout shared by preprocessing and inference."""

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
        if isinstance(point, dict):
            target[base : base + CHANNELS] = (
                point.get("x", 0), point.get("y", 0), point.get("z", 0), point.get("visibility", 1)
            )
        else:
            target[base : base + CHANNELS] = (point.x, point.y, point.z, point.visibility)


def _normalize_group(group: np.ndarray, root_index: int, scale_index: int | None = None) -> None:
    """Normalize coordinates in-place for translation/scale robustness."""
    coords = group[..., :3]
    visible = group[..., 3] > 0
    if not visible.any() or not visible[..., root_index].any():
        return
    raw_coords = coords.copy()
    root = raw_coords[..., root_index : root_index + 1, :]
    if scale_index is not None and scale_index < group.shape[-2] and visible[..., scale_index].any():
        scale = np.linalg.norm(raw_coords[..., scale_index, :] - raw_coords[..., root_index, :], axis=-1, keepdims=True)
    else:
        visible_coords = np.where(visible[..., None], raw_coords, np.nan)
        span = np.nanmax(visible_coords, axis=-2) - np.nanmin(visible_coords, axis=-2)
        scale = np.linalg.norm(span, axis=-1, keepdims=True)
    valid_scale = scale > 1e-3
    if np.any(valid_scale):
        centered = raw_coords - root
        coords[...] = centered / np.where(valid_scale[..., None, :], scale[..., None, :], 1.0)
    group[..., 3] = np.where(visible, group[..., 3], 0).astype(np.float32)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize a [time, 2212] landmark sequence without changing its layout."""
    array = np.asarray(features, dtype=np.float32).copy()
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must have shape [time,{FEATURE_DIM}]")
    landmarks = array.reshape(array.shape[0], 553, 4)
    _normalize_group(landmarks[:, :21], 0, 9)
    _normalize_group(landmarks[:, 21:42], 0, 9)
    _normalize_group(landmarks[:, 42:75], 0, None)
    # Face landmark 1 is a stable central facial anchor in MediaPipe Face Mesh.
    _normalize_group(landmarks[:, 75:], 1, None)
    return array


def observation_to_features(observation: Observation) -> np.ndarray:
    """Map one observation into the versioned fixed feature layout."""
    landmarks = observation.landmarks
    raw = landmarks_to_features(landmarks.left_hand, landmarks.right_hand, landmarks.pose, landmarks.face)
    return normalize_features(raw[None, :])[0]


def landmarks_to_features(left_hand: list | None, right_hand: list | None, pose: list | None, face: list | None) -> np.ndarray:
    """Map MediaPipe-shaped landmark lists into the versioned fixed feature layout."""
    result = np.zeros(FEATURE_DIM, dtype=np.float32)
    hand_size = HAND_POINTS * CHANNELS
    pose_size = POSE_POINTS * CHANNELS
    _append_points(result, 0, left_hand, HAND_POINTS)
    _append_points(result, hand_size, right_hand, HAND_POINTS)
    _append_points(result, hand_size * 2, pose, POSE_POINTS)
    _append_points(result, hand_size * 2 + pose_size, face, FACE_POINTS)
    return result
