"""Shared ranking diagnostics for the MiniLM ESCI cross-encoder."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .evaluation import GAIN, LABELS

EXPECTED_GAIN_ORDER = LABELS  # E, S, C, I

def expected_gain(probabilities: Sequence[float]) -> float:
    """Expected ESCI gain in [0, 3] from probabilities ordered E, S, C, I."""
    return float(sum(p * GAIN[label] for p, label in zip(probabilities, EXPECTED_GAIN_ORDER)))


def validation_summary(rows: Sequence[Mapping[str, Any]], predictions: Sequence[str]) -> dict[str, Any]:
    """Classification metrics on judged rows plus the predicted-label mix on unjudged rows."""
    from collections import Counter

    from .evaluation import classification_metrics

    judged = [(row["label"], pred) for row, pred in zip(rows, predictions) if row["label"] in LABELS]
    unjudged = Counter(pred for row, pred in zip(rows, predictions) if row["label"] not in LABELS)
    metrics = classification_metrics([g for g, _ in judged], [p for _, p in judged])
    confusion = metrics["confusion_matrix"]
    e_row = dict(zip(LABELS, confusion["E"]))
    total_unjudged = sum(unjudged.values())
    metrics["e_to_c"] = e_row["C"]
    metrics["e_to_i"] = e_row["I"]
    metrics["e_to_c_or_i_rate"] = (e_row["C"] + e_row["I"]) / max(1, sum(e_row.values()))
    metrics["unjudged_prediction_counts"] = dict(unjudged)
    metrics["unjudged_predicted_e_or_s_rate"] = (unjudged["E"] + unjudged["S"]) / total_unjudged if total_unjudged else None
    metrics["unjudged_predicted_e_rate"] = unjudged["E"] / total_unjudged if total_unjudged else None
    return metrics
