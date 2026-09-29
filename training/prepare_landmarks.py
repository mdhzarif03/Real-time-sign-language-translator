"""Extract timestamped holistic landmarks and annotated sign boundaries from videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _unwrap(value: Any) -> list:
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        return list(value[0])
    return list(value)


def _point(landmark: Any) -> dict[str, float]:
    return {
        "x": float(landmark.x or 0),
        "y": float(landmark.y or 0),
        "z": float(landmark.z or 0),
        "visibility": float(landmark.visibility if landmark.visibility is not None else 1),
    }


def _boundary_targets(
    timestamps: list[int],
    segments: list[dict[str, Any]],
    vocabulary: dict[str, int],
    sentence_end_ms: int | float,
) -> tuple[list[int], list[int]]:
    if not timestamps:
        raise ValueError("video produced no frames at the selected sample rate")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in timestamps):
        raise ValueError("sample timestamps must be finite numbers")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("sample timestamps must increase strictly")
    if not segments:
        raise ValueError("each continuous sample requires timestamped gloss segments")
    if not isinstance(sentence_end_ms, (int, float)) or isinstance(sentence_end_ms, bool) or not math.isfinite(sentence_end_ms) or sentence_end_ms < 0:
        raise ValueError("each sample requires an annotated sentence_end_ms")
    boundaries = [0] * len(timestamps)
    gloss_ids: list[int] = []
    previous_end = -1
    for segment_index, segment in enumerate(segments):
        gloss = segment.get("gloss")
        start_ms = segment.get("start_ms")
        end_ms = segment.get("end_ms")
        if not isinstance(gloss, str) or gloss not in vocabulary:
            raise ValueError(f"gloss {gloss!r} is missing from the supplied language vocabulary")
        if (
            not isinstance(start_ms, (int, float))
            or not isinstance(end_ms, (int, float))
            or isinstance(start_ms, bool)
            or isinstance(end_ms, bool)
            or not math.isfinite(start_ms)
            or not math.isfinite(end_ms)
            or start_ms < 0
            or end_ms <= start_ms
        ):
            raise ValueError(f"{gloss}: segment times must satisfy 0 <= start_ms < end_ms")
        start_index = min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - start_ms))
        end_index = min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - end_ms))
        start_index = max(start_index, previous_end + 1)
        if start_index >= len(timestamps) or end_index <= start_index:
            raise ValueError(f"{gloss}: segment is too short at the selected sampling rate")
        end_index = min(end_index, len(timestamps) - 1)
        boundaries[start_index] = 1
        for index in range(start_index + 1, end_index):
            boundaries[index] = 2
        boundaries[end_index] = 4 if segment_index == len(segments) - 1 else 3
        previous_end = end_index
        gloss_ids.append(int(vocabulary[gloss]))
    sentence_end_index = min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - sentence_end_ms))
    last_sign_end = min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - segments[-1]["end_ms"]))
    if sentence_end_index != last_sign_end:
        raise ValueError("sentence_end_ms must align with the final annotated gloss end")
    return gloss_ids, boundaries


def _features_from_video(video_path: Path, model_path: Path, sample_hz: float) -> tuple[np.ndarray, list[int]]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError("Install training/requirements.txt in Python 3.12 before extracting video landmarks") from error

    from backend.app.features import FEATURE_DIM, landmarks_to_features

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError(f"video has invalid frame rate: {video_path}")

    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.45,
        min_pose_detection_confidence=0.45,
        min_hand_landmarks_confidence=0.45,
        output_face_blendshapes=False,
    )
    features: list[np.ndarray] = []
    timestamps: list[int] = []
    next_sample_ms = 0.0
    frame_index = 0
    try:
        with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as task:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                timestamp_ms = round(frame_index * 1000 / source_fps)
                frame_index += 1
                if timestamp_ms + 0.5 < next_sample_ms:
                    continue
                next_sample_ms += 1000 / sample_hz
                height, width = frame_bgr.shape[:2]
                scale = min(1.0, 640 / max(width, height))
                if scale < 1:
                    frame_bgr = cv2.resize(frame_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = task.detect_for_video(image, timestamp_ms)
                face = _unwrap(result.face_landmarks)
                pose = _unwrap(result.pose_landmarks)
                left_hand = _unwrap(result.left_hand_landmarks)
                right_hand = _unwrap(result.right_hand_landmarks)
                features.append(landmarks_to_features(
                    [_point(item) for item in left_hand] or None,
                    [_point(item) for item in right_hand] or None,
                    [_point(item) for item in pose] or None,
                    [_point(item) for item in face] or None,
                ))
                timestamps.append(timestamp_ms)
    finally:
        capture.release()
    if not features:
        raise ValueError(f"no sample frames extracted from {video_path}")
    return np.stack(features).astype(np.float32, copy=False).reshape(-1, FEATURE_DIM), timestamps


def _safe_video_path(dataset_root: Path, relative_path: str) -> Path:
    path = (dataset_root / relative_path).resolve()
    if not path.is_relative_to(dataset_root.resolve()):
        raise ValueError("video_path must remain within the dataset root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare(rows: list[dict[str, Any]], dataset_root: Path, output_root: Path, model_path: Path, vocabulary: dict[str, int], sample_hz: float) -> list[dict[str, Any]]:
    if not vocabulary or sorted(vocabulary.values()) != list(range(1, len(vocabulary) + 1)):
        raise ValueError("vocabulary IDs must be contiguous from 1; CTC blank is reserved at 0")
    if sample_hz < 1 or sample_hz > 30:
        raise ValueError("sample_hz must be in [1, 30]")
    signer_splits: dict[str, set[str]] = {}
    session_splits: dict[str, set[str]] = {}
    for row in rows:
        signer_id, session_id, split = row.get("signer_id"), row.get("session_id"), row.get("split")
        if not isinstance(signer_id, str) or not signer_id or not isinstance(session_id, str) or not session_id:
            raise ValueError("every row must include non-empty signer_id and session_id")
        if split not in {"train", "validation", "test"}:
            raise ValueError("every row must have a train, validation or test split")
        signer_splits.setdefault(signer_id, set()).add(split)
        session_splits.setdefault(session_id, set()).add(split)
    if any(len(splits) > 1 for splits in signer_splits.values()):
        raise ValueError("signer leakage across preprocessing splits")
    if any(len(splits) > 1 for splits in session_splits.values()):
        raise ValueError("recording session leakage across preprocessing splits")
    seen: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    (output_root / "features").mkdir(parents=True, exist_ok=True)
    for row in rows:
        sample_id = row.get("sample_id")
        signer_id = row.get("signer_id")
        session_id = row.get("session_id")
        language = row.get("language")
        split = row.get("split")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("sample_id values must be unique non-empty strings")
        if not isinstance(signer_id, str) or not signer_id or not isinstance(session_id, str) or not session_id or split not in {"train", "validation", "test"}:
            raise ValueError(f"{sample_id}: signer_id, session_id and a signer/session-disjoint split are required")
        if not isinstance(language, str) or not language:
            raise ValueError(f"{sample_id}: an explicit sign-language identifier is required")
        seen.add(sample_id)
        features, timestamps = _features_from_video(_safe_video_path(dataset_root, row["video_path"]), model_path, sample_hz)
        gloss_ids, boundaries = _boundary_targets(timestamps, row.get("segments", []), vocabulary, row.get("sentence_end_ms"))
        minimum_ctc_frames = len(gloss_ids) + sum(left == right for left, right in zip(gloss_ids, gloss_ids[1:]))
        if minimum_ctc_frames > len(features):
            raise ValueError(f"{sample_id}: CTC needs at least {minimum_ctc_frames} frames for this gloss sequence")
        name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        feature_path = output_root / "features" / f"{name}.npy"
        boundary_path = output_root / "features" / f"{name}.boundaries.npy"
        np.save(feature_path, features, allow_pickle=False)
        np.save(boundary_path, np.asarray(boundaries, dtype=np.int64), allow_pickle=False)
        output_rows.append({
            "sample_id": sample_id,
            "signer_id": signer_id,
            "session_id": session_id,
            "sequence_id": row.get("sequence_id", sample_id),
            "language": language,
            "feature_file": feature_path.relative_to(output_root).as_posix(),
            "boundary_file": boundary_path.relative_to(output_root).as_posix(),
            "gloss_ids": gloss_ids,
            "split": split,
            "frame_count": len(features),
            "duration_ms": timestamps[-1] - timestamps[0],
        })
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="split JSONL with signer, language, video_path and gloss time segments")
    parser.add_argument("vocabulary", type=Path, help="language vocabulary JSON: gloss string to contiguous integer ID")
    parser.add_argument("dataset_root", type=Path, help="root containing the licensed source videos")
    parser.add_argument("output", type=Path, help="processed data directory")
    parser.add_argument("--model", type=Path, required=True, help="local MediaPipe holistic_landmarker.task asset")
    parser.add_argument("--sample-hz", type=float, default=12.5)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_rows = prepare(rows, args.dataset_root.resolve(), args.output.resolve(), model_path, vocabulary, args.sample_hz)
    manifest_path = args.output / "manifest.features.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    print(f"wrote {len(output_rows)} signer-split feature sequences to {manifest_path}")


if __name__ == "__main__":
    main()
