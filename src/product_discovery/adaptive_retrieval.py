"""Explainable adaptive retrieval decisions for the one search pipeline.

The router is intentionally deterministic.  It does not choose a different
retrieval system: every request still uses BM25, dense retrieval, RRF fusion,
and the same MiniLM cross-encoder.  It only changes the two RRF contributions
from query evidence that a product-search user can understand.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .query_understanding import QueryAnalysis, normalized_tokens


# These are function words and generic shopping words that should not receive a
# title-field bonus.  The regular BM25 score still sees them; this list only
# prevents incidental title overlap from outweighing a brand or model identifier.
FIELD_BOOST_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "pack",
        "set",
        "the",
        "to",
        "with",
    }
)


@dataclass(frozen=True)
class AdaptiveRetrievalDecision:
    """The fixed, per-query RRF weights used by production search."""

    bm25_weight: float
    dense_weight: float
    lexical_rarity: float | None
    reasons: tuple[str, ...]

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FieldLexicalMatch:
    """A bounded field-aware BM25 adjustment for one already-lexical candidate."""

    boost: float
    reasons: tuple[str, ...]


def choose_retrieval_weights(
    analysis: QueryAnalysis, *, lexical_rarity: float | None
) -> AdaptiveRetrievalDecision:
    """Choose one interpretable BM25/dense balance without model routing.

    Higher IDF indicates rarer lexical evidence.  A model identifier takes
    precedence over all other rules, then brand evidence, then long descriptive
    language.  The values are conservative initial operating points rather than
    evaluation-tuned claims and always sum to one.
    """
    rarity = round(float(lexical_rarity), 4) if lexical_rarity is not None else None
    reasons: list[str] = []
    if analysis.model_tokens:
        bm25_weight = 0.82
        reasons.append(f"model token: {', '.join(analysis.model_tokens)}")
    elif analysis.numeric_tokens and (rarity or 0.0) >= 6.0:
        bm25_weight = 0.76
        reasons.append(f"rare numeric token: {', '.join(analysis.numeric_tokens)}")
    elif analysis.brands:
        bm25_weight = 0.68
        reasons.append(f"recognized brand: {', '.join(analysis.brands)}")
    elif (rarity or 0.0) >= 6.0:
        bm25_weight = 0.64
        reasons.append("rare lexical terms")
    elif len(analysis.tokens) >= 6:
        bm25_weight = 0.38
        reasons.append("long natural-language request")
    elif analysis.colors:
        bm25_weight = 0.56
        reasons.append(f"colour attribute: {', '.join(analysis.colors)}")
    else:
        bm25_weight = 0.50
        reasons.append("balanced product query")

    # Negation remains a hybrid request, but preserving a lexical majority makes
    # it more likely that the exclusion adjustment receives the right candidates.
    if analysis.contains_negation and bm25_weight < 0.60:
        bm25_weight = 0.60
        reasons.append("explicit exclusion")
    elif analysis.contains_negation:
        reasons.append("explicit exclusion")

    dense_weight = round(1.0 - bm25_weight, 2)
    return AdaptiveRetrievalDecision(
        bm25_weight=round(bm25_weight, 2),
        dense_weight=dense_weight,
        lexical_rarity=rarity,
        reasons=tuple(reasons),
    )


def field_aware_lexical_match(
    analysis: QueryAnalysis,
    *,
    title: str | None,
    brand: str | None,
    colour: str | None,
) -> FieldLexicalMatch:
    """Return title/brand/colour evidence without rewarding descriptions.

    Description and bullet-point matches remain available in the base BM25
    index, but they deliberately receive no extra score here.  This prevents an
    incidental long-description match from overpowering a product identifier in
    the title or a declared brand.
    """
    title_terms = set(normalized_tokens(title or ""))
    brand_value = " ".join(normalized_tokens(brand or ""))
    colour_value = " ".join(normalized_tokens(colour or ""))
    meaningful_query_terms = {
        token
        for token in analysis.tokens
        if len(token) >= 3
        and token not in FIELD_BOOST_STOPWORDS
        and token not in analysis.brands
        and token not in analysis.colors
        and token not in analysis.model_tokens
    }
    title_matches = sorted(title_terms & meaningful_query_terms)
    matched_models = [token for token in analysis.model_tokens if token in title_terms]

    boost = min(0.75, 0.25 * len(title_matches))
    reasons: list[str] = []
    if title_matches:
        reasons.append(f"title match: {', '.join(title_matches[:3])}")
    if brand_value and brand_value in analysis.brands:
        boost += 1.25
        reasons.append(f"exact brand: {brand}")
    if matched_models:
        boost += min(2.5, 1.25 * len(matched_models))
        reasons.append(f"model token: {', '.join(matched_models)}")
    if colour_value and colour_value in analysis.colors:
        boost += 0.25
        reasons.append(f"colour: {colour}")
    return FieldLexicalMatch(boost=round(boost, 4), reasons=tuple(reasons))


def lexical_fields_match(
    analysis: QueryAnalysis, fields: Mapping[str, str | None]
) -> FieldLexicalMatch:
    """Convenience adapter for the catalog's bounded field lookup."""
    return field_aware_lexical_match(
        analysis,
        title=fields.get("title"),
        brand=fields.get("brand"),
        colour=fields.get("colour"),
    )
