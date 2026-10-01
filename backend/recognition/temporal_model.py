from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from backend.app.features import FEATURE_DIM


@dataclass
class TemporalOutput:
    sign_logits: Tensor
    boundary_logits: Tensor


class LandmarkGraphEncoder(nn.Module):
    """Small visibility-aware graph encoder for one landmark group."""

    def __init__(self, point_count: int, width: int, edges: tuple[tuple[int, int], ...] = ()) -> None:
        super().__init__()
        adjacency = torch.eye(point_count, dtype=torch.float32)
        for left, right in edges:
            adjacency[left, right] = adjacency[right, left] = 1
        degree = adjacency.sum(dim=-1).clamp_min(1).rsqrt()
        self.register_buffer("adjacency", degree[:, None] * adjacency * degree[None, :], persistent=False)
        self.position = nn.Parameter(torch.randn(point_count, width) * 0.01)
        self.input_projection = nn.Linear(4, width)
        self.message = nn.Linear(width, width, bias=False)
        self.update = nn.Linear(width * 2, width)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.Linear(width, 1)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        valid = points[..., 3].gt(0) & torch.isfinite(points[..., :3]).all(dim=-1)
        hidden = self.input_projection(points) + self.position
        neighbors = torch.einsum("ij,btjd->btid", self.adjacency, hidden)
        hidden = self.norm(hidden + torch.tanh(self.update(torch.cat((hidden, self.message(neighbors)), dim=-1))))
        scores = self.attention(hidden).squeeze(-1).masked_fill(~valid, -1e4)
        weights = scores.softmax(dim=-1) * valid.to(hidden.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=-2)
        present = valid.any(dim=-1, keepdim=True).to(hidden.dtype)
        return pooled * present, present


class FaceEncoder(nn.Module):
    """Fast attention pooling for 478 face landmarks; no quadratic face graph."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(4, width)
        self.norm = nn.LayerNorm(width)
        self.score = nn.Linear(width, 1)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        valid = points[..., 3].gt(0) & torch.isfinite(points[..., :3]).all(dim=-1)
        hidden = self.norm(self.projection(points))
        scores = self.score(hidden).squeeze(-1).masked_fill(~valid, -1e4)
        weights = scores.softmax(dim=-1) * valid.to(hidden.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=-2)
        present = valid.any(dim=-1, keepdim=True).to(hidden.dtype)
        return pooled * present, present


class TemporalSignTransformer(nn.Module):
    """Low-latency landmark encoder with a causal GRU and CTC heads.

    The public class name is retained for checkpoint/API compatibility. Version 4
    replaces quadratic temporal attention with a compact GRU and removes the
    quadratic face-landmark message pass, substantially reducing CPU inference.
    """

    def __init__(
        self,
        feature_dim: int,
        vocabulary_size: int,
        width: int = 192,
        heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 384,
        dropout: float = 0.10,
        max_length: int = 2048,
    ) -> None:
        super().__init__()
        if feature_dim != FEATURE_DIM or vocabulary_size <= 0 or width < 64 or layers < 1:
            raise ValueError("invalid model dimensions")
        self.feature_dim = feature_dim
        self.vocabulary_size = vocabulary_size
        branch = width // 2
        self.hand_encoder = LandmarkGraphEncoder(21, branch, (
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        ))
        self.pose_encoder = LandmarkGraphEncoder(33, branch, (
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
            (12, 24), (23, 24), (23, 25), (25, 27), (27, 31), (24, 26),
            (26, 28), (28, 32), (0, 11), (0, 12),
        ))
        self.face_encoder = FaceEncoder(branch)
        fusion_width = width * 2 + 4
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_width), nn.Linear(fusion_width, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.temporal = nn.GRU(
            input_size=width * 2,
            hidden_size=width,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(width)
        self.sign_head = nn.Linear(width, vocabulary_size + 1)
        self.boundary_head = nn.Linear(width, 5)

    def forward(self, features: Tensor, lengths: Tensor) -> TemporalOutput:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape [batch,time,{self.feature_dim}]")
        landmarks = features.reshape(features.shape[0], features.shape[1], 553, 4)
        left_hand, left_present = self.hand_encoder(landmarks[:, :, :21])
        right_hand, right_present = self.hand_encoder(landmarks[:, :, 21:42])
        pose, pose_present = self.pose_encoder(landmarks[:, :, 42:75])
        face, face_present = self.face_encoder(landmarks[:, :, 75:])
        fused = self.fusion(torch.cat((left_hand, right_hand, pose, face, left_present, right_present, pose_present, face_present), dim=-1))
        # Temporal differences make motion explicit without adding a second video model.
        delta = torch.diff(fused, dim=1, prepend=fused[:, :1])
        sequence = torch.cat((fused, delta), dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.temporal(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=features.shape[1])
        encoded = self.temporal_norm(encoded)
        return TemporalOutput(self.sign_head(encoded), self.boundary_head(encoded))


def build_model_from_config(config: dict, vocabulary_size: int) -> TemporalSignTransformer:
    expected_architecture = "spatiotemporal-landmark-ctc-v4-gru"
    if config.get("architecture") != expected_architecture or int(config.get("feature_dim", -1)) != FEATURE_DIM:
        raise ValueError("unsupported temporal model architecture or feature dimension")
    if int(config.get("vocabulary_size", -1)) != vocabulary_size:
        raise ValueError("checkpoint vocabulary size does not match the model configuration")
    width = int(config.get("width", 192))
    layers = int(config.get("layers", 2))
    dropout = float(config.get("dropout", 0.10))
    if not 64 <= width <= 384 or width % 2:
        raise ValueError("checkpoint model width is outside supported limits")
    if not 1 <= layers <= 4 or not 0 <= dropout <= 0.5:
        raise ValueError("checkpoint depth/dropout configuration is outside supported limits")
    return TemporalSignTransformer(FEATURE_DIM, vocabulary_size, width=width, layers=layers, dropout=dropout)


def ctc_objective(output: TemporalOutput, targets: Tensor, target_lengths: Tensor, input_lengths: Tensor, boundary_targets: Tensor) -> Tensor:
    if boundary_targets.shape != output.boundary_logits.shape[:2]:
        raise ValueError("boundary_targets must have shape [batch,time]")
    log_probabilities = output.sign_logits.log_softmax(dim=-1).transpose(0, 1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)(log_probabilities, targets, input_lengths, target_lengths)
    flat_targets = boundary_targets.flatten()
    valid_targets = flat_targets[flat_targets.ne(-100)]
    counts = torch.bincount(valid_targets, minlength=output.boundary_logits.shape[-1]).to(dtype=output.boundary_logits.dtype)
    class_weights = (counts.sum() / (len(counts) * counts.clamp_min(1))).clamp(max=10)
    boundary = nn.functional.cross_entropy(output.boundary_logits.flatten(0, 1), flat_targets, weight=class_weights, ignore_index=-100)
    return ctc + 0.10 * boundary
