"""Evaluation helpers: event matching with a tolerance window, and score metrics.

These functions only compare predictions with labels; they never touch video or models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from smartplay.scoring.automaton import ScoreState, format_score


@dataclass(frozen=True)
class DetectionScore:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_events(
    predicted: Sequence[float], truth: Sequence[float], tolerance: float
) -> DetectionScore:
    """One-to-one matching of event times (or frames) within ``tolerance``.

    Both lists are sorted and swept once; each true event validates at most one prediction,
    and the earliest compatible pair is matched first.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    pred, true = sorted(predicted), sorted(truth)
    i = j = tp = 0
    while i < len(pred) and j < len(true):
        if abs(pred[i] - true[j]) <= tolerance:
            tp += 1
            i += 1
            j += 1
        elif pred[i] < true[j]:
            i += 1
        else:
            j += 1
    return DetectionScore(tp, len(pred) - tp, len(true) - tp)


def point_winner_accuracy(predicted: Sequence[str], truth: Sequence[str]) -> float:
    """Share of points with the correct winner. Sequences must be aligned point by point."""
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must have the same number of points")
    if not truth:
        return 0.0
    return sum(p == t for p, t in zip(predicted, truth)) / len(truth)


def score_timeline_accuracy(
    predicted: Sequence[ScoreState], truth: Sequence[ScoreState]
) -> float:
    """Share of points after which the full displayed score is exactly right."""
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must have the same number of points")
    if not truth:
        return 0.0
    same = sum(format_score(p) == format_score(t) for p, t in zip(predicted, truth))
    return same / len(truth)


def binomial_ci95(successes: int, n: int) -> tuple[float, float]:
    """Wilson 95 % confidence interval for a proportion."""
    if n <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
