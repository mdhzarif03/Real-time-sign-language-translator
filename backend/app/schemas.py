from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Landmark(StrictModel):
    x: float = Field(ge=-2, le=2)
    y: float = Field(ge=-2, le=2)
    z: float = Field(ge=-4, le=4)
    visibility: float = Field(ge=0, le=1)


LandmarkSet = Annotated[list[Landmark] | None, Field(max_length=512)]


class Landmarks(StrictModel):
    left_hand: Annotated[list[Landmark] | None, Field(max_length=21)]
    right_hand: Annotated[list[Landmark] | None, Field(max_length=21)]
    pose: Annotated[list[Landmark] | None, Field(max_length=128)]
    face: LandmarkSet


class Observation(StrictModel):
    schema_version: Literal[1]
    type: Literal["observation"]
    stream_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    sign_language: str = Field(min_length=2, max_length=32)
    image_width: int = Field(default=1280, ge=1, le=8192)
    image_height: int = Field(default=720, ge=1, le=8192)
    landmarks: Landmarks


class Status(StrictModel):
    schema_version: Literal[1] = 1
    type: Literal["status"] = "status"
    state: Literal["ready", "model_loading", "model_unavailable", "camera_unavailable", "degraded", "error"]
    message: str = Field(max_length=512)
    model_id: str | None = Field(default=None, max_length=128)
    model_version: str | None = Field(default=None, max_length=64)


class TokenHypothesis(StrictModel):
    text: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0, le=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    uncertain: bool

    @model_validator(mode="after")
    def valid_time_range(self) -> TokenHypothesis:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class TranslationHypothesis(StrictModel):
    """A language-realizer result, kept separate from the recognized glosses."""

    text: str = Field(min_length=1, max_length=2000)
    language: str = Field(min_length=2, max_length=32)
    confidence: float = Field(ge=0, le=1)
    uncertain: bool


class Hypothesis(StrictModel):
    schema_version: Literal[1] = 1
    type: Literal["hypothesis"] = "hypothesis"
    stream_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=64)
    revision: int = Field(ge=0)
    is_final: bool
    confidence: float = Field(ge=0, le=1)
    tokens: Annotated[list[TokenHypothesis], Field(max_length=256)]
    translation: TranslationHypothesis | None = None
    latency_ms: float | None = Field(default=None, ge=0)
