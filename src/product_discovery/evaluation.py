"""Metrics for held-out ESCI classification and graded product ranking."""
from __future__ import annotations

import math
from collections import Counter

LABELS = ("E", "S", "C", "I")
GAIN = {"E": 3, "S": 2, "C": 1, "I": 0}
ACCEPTABLE_LABELS = {"E", "S"}


def precision_recall_f1(predicted: set[str], expected: set[str]) -> dict[str, float]:
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def classification_metrics(gold: list[str], predicted: list[str]) -> dict:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Non-empty aligned gold and prediction lists required")
    if set(gold + predicted) - set(LABELS):
        raise ValueError("ESCI labels must be one of E, S, C, I")
    confusion = {label: Counter() for label in LABELS}
    for expected, actual in zip(gold, predicted):
        confusion[expected][actual] += 1
    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "support": sum(confusion[label].values())}
    return {
        "examples": len(gold),
        "accuracy": sum(a == b for a, b in zip(gold, predicted)) / len(gold),
        "macro_f1": sum(row["f1"] for row in per_label.values()) / len(LABELS),
        "per_label": per_label,
        "confusion_matrix": {expected: [confusion[expected][actual] for actual in LABELS] for expected in LABELS},
        "confusion_matrix_labels": list(LABELS),
    }


def graded_ndcg(labels: list[str], k: int = 10) -> float:
    gains = [GAIN[label] for label in labels[:k]]
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted((GAIN[label] for label in labels), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(ranked_ids: list[list[str]], relevant_ids: list[set[str]], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    if len(ranked_ids) != len(relevant_ids) or not ranked_ids:
        raise ValueError("Non-empty aligned ranking and relevance lists required")
    output = {f"recall_at_{k}": sum(bool(set(rows[:k]) & expected) for rows, expected in zip(ranked_ids, relevant_ids)) / len(ranked_ids) for k in ks}
    ranks = [next((position + 1 for position, item in enumerate(rows) if item in expected), None) for rows, expected in zip(ranked_ids, relevant_ids)]
    output["mrr"] = sum(1 / rank if rank else 0 for rank in ranks) / len(ranks)
    return output


def ranking_metrics_from_labels(
    ranked_labels: list[list[str]], ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """ESCI ranking metrics using E/S for acceptable products and graded NDCG."""
    if not ranked_labels:
        raise ValueError("At least one ranked query is required")
    if any(set(labels) - set(LABELS) for labels in ranked_labels):
        raise ValueError("ESCI labels must be one of E, S, C, I")
    relevant = [{str(index) for index, label in enumerate(labels) if label in ACCEPTABLE_LABELS} for labels in ranked_labels]
    ranked_ids = [[str(index) for index in range(len(labels))] for labels in ranked_labels]
    metrics = retrieval_metrics(ranked_ids, relevant, ks)
    for k in ks:
        metrics[f"ndcg_at_{k}"] = sum(graded_ndcg(labels, k) for labels in ranked_labels) / len(ranked_labels)
    metrics["queries_with_acceptable_result"] = sum(bool(rows) for rows in relevant)
    return metrics
