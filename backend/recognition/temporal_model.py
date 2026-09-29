from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from backend.app.features import FEATURE_DIM


@dataclass
class TemporalOutput:
    sign_logits: Tensor  # [batch, time, blank + vocabulary]
    boundary_logits: Tensor  # [batch, time, outside/start/inside/end]


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, width: int, max_length: int = 2048) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width))
        encoding = torch.zeros(max_length, width)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence length exceeds configured positional encoding")
        return value + self.encoding[:, : value.shape[1]].to(dtype=value.dtype)


class LandmarkGraphEncoder(nn.Module):
    """Visibility-aware spatial encoder for one anatomical landmark group."""

    def __init__(self, point_count: int, width: int, edges: tuple[tuple[int, int], ...] = ()) -> None:
        super().__init__()
        adjacency = torch.eye(point_count, dtype=torch.float32)
        for left, right in edges:
            adjacency[left, right] = adjacency[right, left] = 1
        degree = adjacency.sum(dim=-1).clamp_min(1).rsqrt()
        adjacency = degree[:, None] * adjacency * degree[None, :]
        self.register_buffer("adjacency", adjacency, persistent=False)
        self.position = nn.Parameter(torch.randn(point_count, width) * 0.02)
        self.input_projection = nn.Linear(4, width)
        self.self_projection = nn.Linear(width, width)
        self.neighbor_projection = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.Linear(width, 1)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        # points: [batch,time,nodes,(x,y,z,visibility)]
        valid = points[..., 3].gt(0) & torch.isfinite(points[..., :3]).all(dim=-1)
        hidden = nn.functional.gelu(self.input_projection(points) + self.position)
        neighbors = torch.einsum("ij,btjd->btid", self.adjacency, hidden)
        hidden = self.norm(hidden + nn.functional.gelu(self.self_projection(hidden) + self.neighbor_projection(neighbors)))
        scores = self.attention(hidden).squeeze(-1).masked_fill(~valid, -1e4)
        weights = scores.softmax(dim=-1) * valid.to(dtype=hidden.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=-2)
        present = valid.any(dim=-1, keepdim=True).to(dtype=hidden.dtype)
        return pooled * present, present


class TemporalSignTransformer(nn.Module):
    """Hand/body/face graph encoder with a temporal CTC gloss and boundary decoder."""

    def __init__(
        self,
        feature_dim: int,
        vocabulary_size: int,
        width: int = 256,
        heads: int = 8,
        layers: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.15,
        max_length: int = 2048,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or vocabulary_size <= 0 or width % heads:
            raise ValueError("invalid model dimensions")
        self.feature_dim = feature_dim
        self.vocabulary_size = vocabulary_size
        if feature_dim != 2212:
            raise ValueError("the spatiotemporal encoder requires the versioned 2,212-value landmark layout")
        self.hand_encoder = LandmarkGraphEncoder(21, width // 2, (
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        ))
        self.pose_encoder = LandmarkGraphEncoder(33, width // 2, (
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
            (12, 24), (23, 24), (23, 25), (25, 27), (27, 31), (24, 26),
            (26, 28), (28, 32), (0, 11), (0, 12),
        ))
        self.face_encoder = LandmarkGraphEncoder(478, width // 2)
        # Each graph branch emits width // 2 features; the four presence flags
        # are appended before the temporal encoder.
        fusion_width = width * 2 + 4
        self.fusion = nn.Sequential(nn.LayerNorm(fusion_width), nn.Linear(fusion_width, width), nn.GELU(), nn.Dropout(dropout))
        self.position = SinusoidalPositionEncoding(width, max_length)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.sign_head = nn.Linear(width, vocabulary_size + 1)  # CTC blank is index zero.
        self.boundary_head = nn.Linear(width, 4)

    def forward(self, features: Tensor, lengths: Tensor) -> TemporalOutput:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape [batch,time,{self.feature_dim}]")
        if lengths.ndim != 1 or lengths.shape[0] != features.shape[0]:
            raise ValueError("lengths must contain one valid-frame count per batch item")
        landmarks = features.reshape(features.shape[0], features.shape[1], 553, 4)
        left_hand, left_present = self.hand_encoder(landmarks[:, :, :21])
        right_hand, right_present = self.hand_encoder(landmarks[:, :, 21:42])
        pose, pose_present = self.pose_encoder(landmarks[:, :, 42:75])
        face, face_present = self.face_encoder(landmarks[:, :, 75:])
        value = self.fusion(torch.cat((left_hand, right_hand, pose, face, left_present, right_present, pose_present, face_present), dim=-1))
        padding_mask = torch.arange(features.shape[1], device=features.device)[None, :] >= lengths[:, None]
        encoded = self.encoder(self.position(value), src_key_padding_mask=padding_mask)
        return TemporalOutput(self.sign_head(encoded), self.boundary_head(encoded))


def build_model_from_config(config: dict, vocabulary_size: int) -> TemporalSignTransformer:
    """Construct only bounded, versioned model configurations from checkpoints."""
    expected_architecture = "spatiotemporal-landmark-ctc-v2"
    if config.get("architecture") != expected_architecture or int(config.get("feature_dim", -1)) != 2212:
        raise ValueError("unsupported temporal model architecture or feature dimension")
    expected_vocabulary_size = int(config.get("vocabulary_size", -1))
    if expected_vocabulary_size != vocabulary_size:
        raise ValueError("checkpoint vocabulary size does not match the model configuration")
    width = int(config.get("width", 256))
    heads = int(config.get("heads", 8))
    layers = int(config.get("layers", 4))
    feedforward_dim = int(config.get("feedforward_dim", 768))
    dropout = float(config.get("dropout", 0.15))
    if not (32 <= width <= 512 and 1 <= heads <= 16 and width % heads == 0):
        raise ValueError("checkpoint transformer width/head configuration is outside supported limits")
    if not 1 <= layers <= 8 or not width <= feedforward_dim <= 2048 or not 0 <= dropout <= 0.5:
        raise ValueError("checkpoint transformer depth/feedforward/dropout configuration is outside supported limits")
    return TemporalSignTransformer(
        FEATURE_DIM,
        vocabulary_size,
        width=width,
        heads=heads,
        layers=layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
    )


def ctc_objective(
    output: TemporalOutput,
    targets: Tensor,
    target_lengths: Tensor,
    input_lengths: Tensor,
    boundary_targets: Tensor,
) -> Tensor:
    if boundary_targets.shape != output.boundary_logits.shape[:2]:
        raise ValueError("boundary_targets must have shape [batch,time]")
    log_probabilities = output.sign_logits.log_softmax(dim=-1).transpose(0, 1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)(log_probabilities, targets, input_lengths, target_lengths)
    boundary = nn.functional.cross_entropy(
        output.boundary_logits.flatten(0, 1), boundary_targets.flatten(), ignore_index=-100
    )
    return ctc + 0.15 * boundary
