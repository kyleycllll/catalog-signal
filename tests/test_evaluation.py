from product_discovery.evaluation import (
    classification_metrics,
    graded_ndcg,
    precision_recall_f1,
    ranking_metrics_from_labels,
    retrieval_metrics,
)


def test_intent_attribute_metrics():
    assert precision_recall_f1({"black", "waterproof"}, {"black", "casual"})["f1"] == .5


def test_retrieval_metrics():
    result = retrieval_metrics([["a", "b"], ["z", "c"]], [{"a"}, {"c"}])
    assert result["recall_at_1"] == .5 and result["recall_at_5"] == 1.0 and result["mrr"] == .75


def test_esci_classification_and_graded_ndcg():
    metrics = classification_metrics(["E", "S", "C", "I"], ["E", "S", "I", "I"])
    assert metrics["accuracy"] == .75 and metrics["macro_f1"] < 1
    assert graded_ndcg(["E", "S", "C", "I"]) == 1


def test_label_ranking_metrics_treat_exact_and_substitute_as_acceptable():
    metrics = ranking_metrics_from_labels([["C", "S", "I"], ["E", "I"]])
    assert metrics["recall_at_1"] == .5
    assert metrics["mrr"] == .75
    assert 0 < metrics["ndcg_at_1"] < 1
