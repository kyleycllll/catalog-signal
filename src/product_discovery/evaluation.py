"""Metrics for held-out ESCI classification and graded product ranking."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

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
    """Compute NDCG for a fully judged, already-ranked candidate pool."""
    _validate_labels(labels)
    _validate_k(k)
    gains = [GAIN[label] for label in labels[:k]]
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted((GAIN[label] for label in labels), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be positive")


def _validate_labels(labels: Sequence[str]) -> None:
    if set(labels) - set(LABELS):
        raise ValueError("ESCI labels must be one of E, S, C, I")


def hit_rate_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Return 1 when at least one known relevant ID appears in the original top K."""
    _validate_k(k)
    return float(bool(set(ranked_ids[:k]) & relevant_ids))


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Return the fraction of all known relevant IDs retrieved in the original top K.

    This is a true per-query Recall@K only when ``relevant_ids`` is the complete
    relevance set for the query. With ESCI's partial labels, callers must describe
    this as recall over *known judged positives*, not catalog-wide recall.
    """
    _validate_k(k)
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Return relevant IDs divided by returned positions, up to K.

    The denominator is ``min(k, len(ranked_ids))`` so short result lists are not
    penalized as though missing positions had been returned. As with Recall@K, this
    is a full precision metric only when relevance judgments are complete.
    """
    _validate_k(k)
    denominator = min(k, len(ranked_ids))
    if not denominator:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / denominator


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Return reciprocal rank using positions from the original candidate ranking."""
    for position, product_id in enumerate(ranked_ids, start=1):
        if product_id in relevant_ids:
            return 1.0 / position
    return 0.0


def retrieval_metrics(
    ranked_ids: list[list[str]], relevant_ids: list[set[str]], ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Aggregate candidate-retrieval metrics without changing ranking positions.

    ``relevant_ids`` must be complete to interpret Recall@K and Precision@K as
    catalog-wide metrics. The experiment runner calls these observed coverage
    metrics because ESCI judgments are incomplete for a full catalog.
    """
    if len(ranked_ids) != len(relevant_ids) or not ranked_ids:
        raise ValueError("Non-empty aligned ranking and relevance lists required")
    for k in ks:
        _validate_k(k)
    output: dict[str, float] = {}
    for k in ks:
        output[f"hit_rate_at_{k}"] = sum(
            hit_rate_at_k(rows, expected, k) for rows, expected in zip(ranked_ids, relevant_ids)
        ) / len(ranked_ids)
        output[f"recall_at_{k}"] = sum(
            recall_at_k(rows, expected, k) for rows, expected in zip(ranked_ids, relevant_ids)
        ) / len(ranked_ids)
        output[f"precision_at_{k}"] = sum(
            precision_at_k(rows, expected, k) for rows, expected in zip(ranked_ids, relevant_ids)
        ) / len(ranked_ids)
    output["mrr"] = sum(
        reciprocal_rank(rows, expected) for rows, expected in zip(ranked_ids, relevant_ids)
    ) / len(ranked_ids)
    return output


def judged_precision_at_k(ranked_labels: Sequence[str], k: int = 10) -> float:
    """Precision@K for an explicitly supplied, fully judged candidate pool.

    The input must already represent the judged pool being evaluated. This helper
    deliberately does not remove unknown documents from a full-catalog ranking.
    """
    _validate_labels(ranked_labels)
    _validate_k(k)
    denominator = min(k, len(ranked_labels))
    if not denominator:
        return 0.0
    return sum(label in ACCEPTABLE_LABELS for label in ranked_labels[:k]) / denominator


def judged_pool_ndcg_at_k(ranked_labels: Sequence[str], k: int = 10) -> float:
    """NDCG@K over an explicitly supplied, fully judged candidate pool only."""
    _validate_labels(ranked_labels)
    if not ranked_labels:
        return 0.0
    return graded_ndcg(list(ranked_labels), k)


def judged_pool_mrr(ranked_labels: Sequence[str]) -> float:
    """MRR over an explicitly supplied, fully judged candidate pool only."""
    _validate_labels(ranked_labels)
    for position, label in enumerate(ranked_labels, start=1):
        if label in ACCEPTABLE_LABELS:
            return 1.0 / position
    return 0.0


def ranking_metrics_from_labels(
    ranked_labels: list[list[str]], ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Aggregate metrics for explicitly supplied fully judged candidate pools.

    This function is intentionally unsuitable for a full ranking containing
    unjudged documents. Callers must use an explicit `judged_pool` name in their
    reports and keep end-to-end candidate coverage separate.
    """
    for labels in ranked_labels:
        _validate_labels(labels)
    for k in ks:
        _validate_k(k)
    if not ranked_labels:
        return {
            **{f"judged_precision_at_{k}": 0.0 for k in ks},
            **{f"judged_pool_ndcg_at_{k}": 0.0 for k in ks},
            "judged_pool_mrr": 0.0,
            "queries_with_acceptable_result": 0,
        }
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"judged_precision_at_{k}"] = sum(
            judged_precision_at_k(labels, k) for labels in ranked_labels
        ) / len(ranked_labels)
        metrics[f"judged_pool_ndcg_at_{k}"] = sum(
            judged_pool_ndcg_at_k(labels, k) for labels in ranked_labels
        ) / len(ranked_labels)
    metrics["judged_pool_mrr"] = sum(
        judged_pool_mrr(labels) for labels in ranked_labels
    ) / len(ranked_labels)
    metrics["queries_with_acceptable_result"] = sum(
        any(label in ACCEPTABLE_LABELS for label in labels) for labels in ranked_labels
    )
    return metrics


def _dcg(gains: Sequence[int], k: int) -> float:
    return sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains[:k]))


def ndcg_unjudged_as_zero(
    ranked_ids: Sequence[str],
    labels: dict[str, str],
    k: int,
    ideal_pool: Sequence[str] | None = None,
) -> float:
    """Full-rank NDCG@K over an original ranking with partial ESCI judgments.

    Every ranked position is kept; products without a judgment for the query get
    gain 0 (so the score is a lower bound). The ideal DCG is computed from the
    judged labels of ``ideal_pool`` when given (e.g. a fixed candidate set), or
    from *all* known judgments for the query otherwise.
    """
    _validate_k(k)
    _validate_labels(list(labels.values()))
    gains = [GAIN[labels[pid]] if pid in labels else 0 for pid in ranked_ids]
    if ideal_pool is None:
        ideal = sorted((GAIN[label] for label in labels.values()), reverse=True)
    else:
        ideal = sorted((GAIN[labels[pid]] if pid in labels else 0 for pid in ideal_pool), reverse=True)
    idcg = _dcg(ideal, k)
    return _dcg(gains, k) / idcg if idcg else 0.0


def judged_pool_labels(ranked_ids: Sequence[str], labels: dict[str, str]) -> list[str]:
    """Condense a ranking to its judged items, preserving relative order (diagnostic only)."""
    return [labels[pid] for pid in ranked_ids if pid in labels]
