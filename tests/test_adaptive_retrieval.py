from product_discovery.adaptive_retrieval import (
    choose_retrieval_weights,
    field_aware_lexical_match,
)
from product_discovery.query_understanding import analyze_query


def test_model_number_query_strongly_favours_lexical_retrieval():
    decision = choose_retrieval_weights(
        analyze_query("makita dtd172 impact drill"), lexical_rarity=8.1
    )

    assert (decision.bm25_weight, decision.dense_weight) == (0.82, 0.18)
    assert decision.reasons == ("model token: dtd172",)


def test_long_natural_language_query_favours_dense_retrieval():
    decision = choose_retrieval_weights(
        analyze_query("comfortable shoes for standing all day"), lexical_rarity=2.5
    )

    assert (decision.bm25_weight, decision.dense_weight) == (0.38, 0.62)
    assert decision.reasons == ("long natural-language request",)


def test_negation_keeps_a_lexical_majority_and_remains_hybrid():
    decision = choose_retrieval_weights(analyze_query("cups without lid"), lexical_rarity=1.2)

    assert (decision.bm25_weight, decision.dense_weight) == (0.60, 0.40)
    assert decision.reasons == ("balanced product query", "explicit exclusion")


def test_field_aware_match_rewards_brand_title_model_and_colour_not_description():
    analysis = analyze_query("makita dtd172 black impact drill")
    match = field_aware_lexical_match(
        analysis,
        title="Makita DTD-172 Impact Driver",
        brand="Makita",
        colour="Black",
    )

    assert match.boost == 3.0
    assert match.reasons == (
        "title match: impact",
        "exact brand: Makita",
        "model token: dtd172",
        "colour: Black",
    )
