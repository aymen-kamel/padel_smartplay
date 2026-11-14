"""Typed records exchanged between pipeline stages.

Every record that comes from perception carries a confidence in [0, 1]. Positions are in the
world frame defined in ``smartplay.geometry.court`` unless a field name ends in ``_uv`` (pixels).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Team = Literal["A", "B"]
Side = Literal["NEAR", "FAR"]
EventKind = Literal["hit", "floor_bounce", "wall", "net", "serve"]
Visibility = Literal["visible", "occluded", "inferred", "lost"]
EndReason = Literal["double_bounce", "wall_or_out", "net", "fault", "other"]
Confidence = Field(ge=0.0, le=1.0)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VideoInfo(Record):
    path: str
    fps: float = Field(gt=0)
    n_frames: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    has_audio: bool

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps


class PlayerTrack(Record):
    frame: int = Field(ge=0)
    player_id: Literal[1, 2, 3, 4]
    team: Team
    side: Side
    foot_xy_m: tuple[float, float]
    conf: float = Confidence
    observed: bool = True  # False when the position is predicted, not detected


class BallState(Record):
    frame: int = Field(ge=0)
    uv: tuple[float, float] | None = None
    xyz_m: tuple[float, float, float] | None = None
    visibility: Visibility
    conf: float = Confidence


class Event(Record):
    t: float = Field(ge=0)
    frame: int = Field(ge=0)
    kind: EventKind
    side: Side | None = None
    player_id: Literal[1, 2, 3, 4] | None = None
    xy_m: tuple[float, float] | None = None
    conf: float = Confidence
    sources: tuple[Literal["audio", "trajectory", "pose", "label"], ...] = ()


class Rally(Record):
    point_idx: int = Field(ge=0)
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)
    server_team: Team | None = None
    conf: float = Confidence


class PointOutcome(Record):
    """Who won a point, as a probability. 1.0 means team A certainly won."""

    point_idx: int = Field(ge=0)
    p_team_a: float = Confidence
    reason: EndReason | None = None

    @property
    def likely_winner(self) -> Team:
        return "A" if self.p_team_a >= 0.5 else "B"


class PointLabel(Record):
    """Ground-truth label for one point (labelling format L1 in docs/architecture/PLAN.md)."""

    match: str
    point_idx: int = Field(ge=0)
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)
    server_team: Team
    winner_team: Team
    end_reason: EndReason
    score_after: str
