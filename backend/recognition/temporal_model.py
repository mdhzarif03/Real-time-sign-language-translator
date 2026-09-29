from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


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


class TemporalSignTransformer(nn.Module):
    """CTC gloss encoder with an auxiliary boundary head; not a language model."""

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
        self.projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width), nn.GELU(), nn.Dropout(dropout))
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
        padding_mask = torch.arange(features.shape[1], device=features.device)[None, :] >= lengths[:, None]
        encoded = self.encoder(self.position(self.projection(features)), src_key_padding_mask=padding_mask)
        return TemporalOutput(self.sign_head(encoded), self.boundary_head(encoded))


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
