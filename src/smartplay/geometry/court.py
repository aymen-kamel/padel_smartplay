"""Padel court constants and the single coordinate convention of the project.

World frame (metres):
    origin  centre of the net, on the floor
    x       across the court, -5 .. +5, positive to the camera's right
    y       along the court, -10 .. +10; negative = NEAR the camera, positive = FAR
    z       up, 0 = floor

Dimensions follow the FIP Rules of Padel (review applicable 01.01.2026), section "The Court".
"""

from __future__ import annotations

from typing import Literal

Side = Literal["NEAR", "FAR"]

COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
HALF_WIDTH_M = COURT_WIDTH_M / 2
HALF_LENGTH_M = COURT_LENGTH_M / 2
SERVICE_LINE_FROM_NET_M = 6.95
NET_HEIGHT_CENTRE_M = 0.88
NET_HEIGHT_ENDS_M = 0.92
BACK_WALL_HEIGHT_M = 3.0
BACK_ENCLOSURE_HEIGHT_M = 4.0
BALL_DIAMETER_MIN_M = 0.0635
BALL_DIAMETER_MAX_M = 0.0677


def side_of(y_m: float) -> Side:
    """Court half containing a point. The net line itself (y == 0) counts as FAR."""
    return "NEAR" if y_m < 0 else "FAR"


def other_side(side: Side) -> Side:
    return "FAR" if side == "NEAR" else "NEAR"


def is_in_court(x_m: float, y_m: float, margin_m: float = 0.0) -> bool:
    """True if the floor point lies inside the court. Lines are in, so use margin >= 0."""
    return abs(x_m) <= HALF_WIDTH_M + margin_m and abs(y_m) <= HALF_LENGTH_M + margin_m


def is_in_service_box(
    x_m: float, y_m: float, side: Side, box: Literal["left", "right"], margin_m: float = 0.0
) -> bool:
    """True if the floor point is in a service box of the given court half.

    ``box`` is named from the camera's point of view: "left" is x < 0, "right" is x > 0.
    The box spans from the net to the service line. Lines count as good (FIP Rule 7.1.c).
    """
    if side_of(y_m) != side and y_m != 0:
        return False
    if abs(y_m) > SERVICE_LINE_FROM_NET_M + margin_m:
        return False
    if abs(x_m) > HALF_WIDTH_M + margin_m:
        return False
    return x_m <= margin_m if box == "left" else x_m >= -margin_m
