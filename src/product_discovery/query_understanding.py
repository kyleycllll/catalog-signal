"""Deterministic query normalization for product search.

This module intentionally contains no learned model or network call.  It turns a
raw customer query into one stable representation used by retrieval and another
that retains exclusions for ranking.  The small brand and colour lexicons are
deliberately conservative: an unknown token remains a normal query token rather
than being guessed as an attribute.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable


# These are common product-search brands, plus the brands used in documented
# examples.  Callers can extend the supported set with catalog-specific names.
DEFAULT_BRANDS = frozenset(
    {
        "adidas",
        "apple",
        "asus",
        "black+decker",
        "bose",
        "canon",
        "dewalt",
        "hp",
        "levi's",
        "makita",
        "new balance",
        "nike",
        "samsung",
        "sony",
        "tervis",
    }
)

COLOURS = frozenset(
    {
        "beige",
        "black",
        "blue",
        "bronze",
        "brown",
        "clear",
        "gold",
        "gray",
        "green",
        "grey",
        "ivory",
        "navy",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "tan",
        "teal",
        "white",
        "yellow",
    }
)

# Long Unicode dashes are separators (``Makita—DTD-172``), while an ASCII
# hyphen remains available for compact spelling/model normalization below.
_DASHES = str.maketrans({character: " " for character in "‐‑‒–—―−"})
_COMPACT_ALIASES = {"airmax": ("air", "max")}
_NEGATION_CUES = frozenset({"without", "excluding", "exclude", "except", "no", "not"})
_EXCLUSION_BOUNDARIES = frozenset({"and", "but", "for", "with", "or"})


@dataclass(frozen=True)
class QueryAnalysis:
    original_query: str
    normalized_query: str
    retrieval_query: str
    tokens: tuple[str, ...]
    brands: tuple[str, ...]
    colors: tuple[str, ...]
    model_tokens: tuple[str, ...]
    numeric_tokens: tuple[str, ...]
    exclusions: tuple[str, ...]
    contains_negation: bool

    def model_dump(self) -> dict:
        """Return JSON-friendly lists for FastAPI responses and trace records."""
        return asdict(self)


def _normalized_tokens(query: str) -> list[str]:
    """Normalize Unicode, punctuation, hyphens and whitespace to ASCII tokens."""
    value = unicodedata.normalize("NFKC", query).casefold().translate(_DASHES)
    # Product model tokens are frequently written with optional hyphens, e.g.
    # DTD-172.  Joining alphanumeric neighbours preserves that identifier.
    value = re.sub(r"(?<=[\w])-(?=[\w])", "", value)
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    value = re.sub(r"[^a-z0-9]+", " ", value)
    raw_tokens = value.split()
    expanded: list[str] = []
    for token in raw_tokens:
        expanded.extend(_COMPACT_ALIASES.get(token, (token,)))
    return expanded


def normalized_tokens(value: str) -> list[str]:
    """Public, deterministic field normalizer shared by lexical feature checks."""
    return _normalized_tokens(value)


def _find_exclusions(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Split searchable product terms from terms following a negation cue."""
    searchable: list[str] = []
    exclusions: list[str] = []
    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if token not in _NEGATION_CUES:
            searchable.append(token)
            cursor += 1
            continue
        cursor += 1
        excluded: list[str] = []
        while cursor < len(tokens) and tokens[cursor] not in _EXCLUSION_BOUNDARIES:
            if tokens[cursor] not in _NEGATION_CUES:
                excluded.append(tokens[cursor])
            cursor += 1
        exclusions.extend(excluded)
        # "without lid and straw" represents two exclusions; keeping the
        # conjunction out of the main query gives a more useful product query.
        if cursor < len(tokens) and tokens[cursor] in {"and", "or"}:
            cursor += 1
    return searchable, list(dict.fromkeys(exclusions))


def _brand_tokens(brands: Iterable[str]) -> dict[tuple[str, ...], str]:
    output: dict[tuple[str, ...], str] = {}
    for brand in brands:
        pieces = tuple(_normalized_tokens(brand))
        if pieces:
            output[pieces] = " ".join(pieces)
    return output


def analyze_query(query: str, supported_brands: Iterable[str] = DEFAULT_BRANDS) -> QueryAnalysis:
    """Return deterministic product-search features for ``query``.

    ``normalized_query`` keeps the user's complete normalized wording so the
    cross-encoder can see a phrase such as "without lid".  ``retrieval_query``
    removes the exclusion phrase so lexical and dense candidate generation focus
    on the desired product itself.
    """
    original = query.strip()
    normalized_tokens = _normalized_tokens(original)
    searchable, exclusions = _find_exclusions(normalized_tokens)
    searchable = searchable or normalized_tokens
    brand_map = _brand_tokens(set(DEFAULT_BRANDS) | set(supported_brands))
    brands: list[str] = []
    for start in range(len(searchable)):
        for length in range(min(3, len(searchable) - start), 0, -1):
            brand = brand_map.get(tuple(searchable[start : start + length]))
            if brand and brand not in brands:
                brands.append(brand)
    colors = [token for token in searchable if token in COLOURS]
    # Letter+digit identifiers (DTD172, A12) and long digit runs are stronger
    # lexical identifiers than ordinary language tokens.
    model_tokens = [
        token
        for token in searchable
        if (re.fullmatch(r"(?=.*[a-z])(?=.*\d)[a-z0-9]+", token) or re.fullmatch(r"\d{3,}", token))
    ]
    numeric_tokens = [token for token in searchable if any(character.isdigit() for character in token)]
    return QueryAnalysis(
        original_query=original,
        normalized_query=" ".join(normalized_tokens),
        retrieval_query=" ".join(searchable),
        tokens=tuple(searchable),
        brands=tuple(brands),
        colors=tuple(dict.fromkeys(colors)),
        model_tokens=tuple(dict.fromkeys(model_tokens)),
        numeric_tokens=tuple(dict.fromkeys(numeric_tokens)),
        exclusions=tuple(exclusions),
        contains_negation=bool(exclusions),
    )
