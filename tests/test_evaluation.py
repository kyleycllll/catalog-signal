from product_discovery.evaluation import (
    classification_metrics,
    graded_ndcg,
    hit_rate_at_k,
    judged_pool_mrr,
    judged_pool_ndcg_at_k,
    judged_precision_at_k,
    precision_recall_f1,
    precision_at_k,
    recall_at_k,
    ranking_metrics_from_labels,
    reciprocal_rank,
    retrieval_metrics,
)


def test_intent_attribute_metrics():
    assert precision_recall_f1({"black", "waterproof"}, {"black", "casual"})["f1"] == .5


def test_retrieval_metrics_distinguish_hit_rate_recall_precision_and_original_rank_mrr():
    ranked = ["X", "A", "Y", "B", "Z"]
    relevant = {"A", "B", "C"}
    assert hit_rate_at_k(ranked, relevant, 5) == 1.0
    assert recall_at_k(ranked, relevant, 5) == 2 / 3
    assert precision_at_k(ranked, relevant, 5) == 2 / 5
    assert reciprocal_rank(ranked, relevant) == 1 / 2

    result = retrieval_metrics([ranked], [relevant], ks=(5,))
    assert result == {
        "hit_rate_at_5": 1.0,
        "recall_at_5": 2 / 3,
        "precision_at_5": 2 / 5,
        "mrr": 1 / 2,
    }


def test_retrieval_metric_edge_cases():
    relevant = {"A", "B", "C"}
    assert hit_rate_at_k(["X", "Y"], relevant, 5) == 0.0
    assert recall_at_k(["X", "Y"], relevant, 5) == 0.0
    assert precision_at_k(["X", "Y"], relevant, 5) == 0.0
    assert reciprocal_rank(["X", "Y"], relevant) == 0.0

    assert hit_rate_at_k(["A", "B", "C"], relevant, 5) == 1.0
    assert recall_at_k(["A", "B", "C"], relevant, 5) == 1.0
    assert precision_at_k(["A", "B", "C"], relevant, 5) == 1.0

    # A short list uses returned positions as its precision denominator.
    assert precision_at_k(["A"], relevant, 5) == 1.0
    assert recall_at_k(["A"], relevant, 5) == 1 / 3
    # MRR retains the original position instead of filtering unknown items.
    assert reciprocal_rank(["unknown-1", "unknown-2", "A"], relevant) == 1 / 3


def test_esci_classification_and_graded_ndcg():
    metrics = classification_metrics(["E", "S", "C", "I"], ["E", "S", "I", "I"])
    assert metrics["accuracy"] == .75 and metrics["macro_f1"] < 1
    assert graded_ndcg(["E", "S", "C", "I"]) == 1


def test_label_ranking_metrics_treat_exact_and_substitute_as_acceptable():
    metrics = ranking_metrics_from_labels([["C", "S", "I"], ["E", "I"]])
    assert metrics["judged_precision_at_1"] == .5
    assert metrics["judged_pool_mrr"] == .75
    assert 0 < metrics["judged_pool_ndcg_at_1"] < 1


def test_judged_pool_metrics_are_explicit_and_handle_no_judged_products():
    labels = ["C", "S", "I"]
    assert judged_precision_at_k(labels, 5) == 1 / 3
    assert judged_pool_mrr(labels) == 1 / 2
    assert 0 < judged_pool_ndcg_at_k(labels, 1) < 1

    assert judged_precision_at_k([], 5) == 0.0
    assert judged_pool_mrr([]) == 0.0
    assert judged_pool_ndcg_at_k([], 5) == 0.0
    empty = ranking_metrics_from_labels([])
    assert empty["judged_precision_at_5"] == 0.0
    assert empty["judged_pool_ndcg_at_10"] == 0.0
    assert empty["judged_pool_mrr"] == 0.0


def test_ndcg_unjudged_as_zero_keeps_positions_and_uses_explicit_ideal():
    from product_discovery.evaluation import judged_pool_labels, ndcg_unjudged_as_zero

    labels = {"a": "E", "b": "S", "c": "I", "z": "E"}
    ranked = ["u1", "a", "b", "c"]  # u1 unjudged at rank 1
    full_ideal = ndcg_unjudged_as_zero(ranked, labels, 3)
    candidate_ideal = ndcg_unjudged_as_zero(ranked, labels, 3, ideal_pool=ranked)
    # The unjudged item at rank 1 costs position; condensing would hide that.
    assert 0 < full_ideal < candidate_ideal < 1
    assert ndcg_unjudged_as_zero(["a", "b", "u1", "c"], labels, 3, ideal_pool=ranked) == 1.0
    assert judged_pool_labels(ranked, labels) == ["E", "S", "I"]
    assert judged_pool_ndcg_at_k(judged_pool_labels(ranked, labels), 3) == 1.0
    assert ndcg_unjudged_as_zero(["u1", "u2"], {"c": "I"}, 5) == 0.0
