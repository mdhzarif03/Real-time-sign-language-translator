from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from backend.app.features import FEATURE_DIM
from backend.app.schemas import Hypothesis, Status, TokenHypothesis
from .temporal_model import TemporalSignTransformer

FEATURE_LAYOUT = "hands-left-right-21x4_pose-33x4_face-478x4_v1"


class TemporalRuntime:
    def __init__(self, model, vocabulary: dict[str, int], language: str, model_id: str, version: str, device: torch.device, window_size: int):
        self.model = model.eval()
        self.vocabulary = {int(index): label for label, index in vocabulary.items()}
        self.language = language
        self.model_id = model_id
        self.version = version
        self.device = device
        self.window_size = window_size
        self.lock = threading.Lock()

    @classmethod
    def load(cls, manifest_path: str | Path) -> tuple[TemporalRuntime | None, Status]:
        path = Path(manifest_path).expanduser().resolve()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            required = ("model_id", "model_version", "sign_language", "checkpoint", "feature_layout")
            if not isinstance(manifest, dict) or any(not isinstance(manifest.get(key), str) or not manifest[key] for key in required):
                raise ValueError("manifest is missing required string fields")
            if manifest["feature_layout"] != FEATURE_LAYOUT:
                raise ValueError("checkpoint feature layout is incompatible")
            checkpoint_path = (path.parent / manifest["checkpoint"]).resolve()
            if not checkpoint_path.is_relative_to(path.parent.resolve()):
                raise ValueError("checkpoint path must remain within the manifest directory")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("sign_language") != manifest["sign_language"]:
                raise ValueError("checkpoint sign language does not match the manifest")
            if checkpoint.get("feature_layout") != FEATURE_LAYOUT:
                raise ValueError("checkpoint feature layout does not match runtime")
            vocabulary = checkpoint.get("vocabulary")
            config = checkpoint.get("model_config")
            if not isinstance(checkpoint, dict) or not isinstance(vocabulary, dict) or not isinstance(config, dict):
                raise ValueError("checkpoint is missing vocabulary or model configuration")
            if int(config.get("feature_dim", -1)) != FEATURE_DIM or int(config.get("vocabulary_size", -1)) != len(vocabulary):
                raise ValueError("checkpoint dimensions do not match the configured vocabulary/features")
            ids = sorted(int(index) for index in vocabulary.values())
            if ids != list(range(1, len(vocabulary) + 1)):
                raise ValueError("checkpoint vocabulary IDs must be contiguous from 1")
            model = TemporalSignTransformer(FEATURE_DIM, len(vocabulary))
            model.load_state_dict(checkpoint["model_state"], strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            window_size = int(manifest.get("window_size", 96))
            if not 8 <= window_size <= 512:
                raise ValueError("window_size must be between 8 and 512 observations")
            runtime = cls(model, vocabulary, manifest["sign_language"], manifest["model_id"], manifest["model_version"], device, window_size)
            status = Status(state="ready", message=f"{manifest['sign_language']} recognition model loaded locally.", model_id=manifest["model_id"], model_version=manifest["model_version"])
            return runtime, status
        except Exception as error:
            return None, Status(state="model_unavailable", message=f"Local recognition model could not be loaded: {error}")

    def status_for(self, language: str | None = None) -> Status:
        if language and language != self.language:
            return Status(state="model_unavailable", message=f"No model is installed for {language}.")
        return Status(state="ready", message=f"{self.language} recognition model is running locally.", model_id=self.model_id, model_version=self.version)

    @torch.inference_mode()
    def predict(self, observations: Sequence[tuple[np.ndarray, int]], stream_id: str, revision: int) -> Hypothesis | None:
        if len(observations) < 8:
            return None
        selected = observations[-self.window_size :]
        features = np.stack([item[0] for item in selected]).astype(np.float32, copy=False)
        timestamps = [item[1] for item in selected]
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=self.device)
        with self.lock:
            output = self.model(tensor, lengths)
        probabilities = output.sign_logits[0].softmax(dim=-1)
        labels = output.sign_logits[0].argmax(dim=-1).tolist()
        tokens: list[TokenHypothesis] = []
        last_label = 0
        active_id = 0
        active_start = 0
        active_scores: list[float] = []
        active_end = 0

        def flush() -> None:
            if active_id == 0 or not active_scores:
                return
            confidence = float(sum(active_scores) / len(active_scores))
            tokens.append(TokenHypothesis(
                text=self.vocabulary.get(active_id, "[unknown]"),
                confidence=confidence,
                start_ms=timestamps[active_start],
                end_ms=timestamps[active_end],
                uncertain=confidence < 0.70,
            ))

        for index, label in enumerate(labels):
            if label == active_id and label != 0:
                active_scores.append(float(probabilities[index, label].item()))
                active_end = index
            elif label != 0 and label != last_label:
                flush()
                active_id = label
                active_start = active_end = index
                active_scores = [float(probabilities[index, label].item())]
            elif label == 0:
                flush()
                active_id = 0
                active_scores = []
            last_label = label
        flush()

        boundary = output.boundary_logits[0, -min(4, len(selected)) :].softmax(dim=-1)
        is_final = bool((boundary[:, 3].mean() >= 0.65).item())
        confidence = sum(token.confidence for token in tokens) / max(1, len(tokens))
        return Hypothesis(
            stream_id=stream_id,
            model_id=self.model_id,
            model_version=self.version,
            revision=revision,
            is_final=is_final,
            confidence=confidence,
            tokens=tokens,
        )
