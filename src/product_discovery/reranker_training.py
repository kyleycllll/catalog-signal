"""Shared pieces for training/evaluating ESCI rerankers (Qwen QLoRA and cross-encoder).

The Qwen training example is exactly the serving condition: the prompt from
``local_reranker.classification_prompt`` tokenized on its own, followed by the
target ``{"label":"X"}`` and EOS. Loss is on the target tokens only. The label is
never truncated away: if prompt + target exceed ``max_length``, the description
and then the bullet points are shortened until they fit.
"""
from __future__ import annotations

import json
import random
from typing import Any, Mapping, Sequence

from .evaluation import GAIN, LABELS
from .local_reranker import classification_prompt

EXPECTED_GAIN_ORDER = LABELS  # E, S, C, I


def target_text(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Not an ESCI label: {label}")
    return json.dumps({"label": label}, separators=(",", ":"))


def fit_prompt_ids(row: Mapping[str, Any], tokenizer, max_length: int, reserved: int) -> tuple[list[int], bool]:
    """Token IDs of the serving prompt, shortening product text so ``reserved`` more tokens fit."""
    fields = dict(row)
    shortened = False
    for _ in range(64):
        ids = tokenizer(classification_prompt(fields), add_special_tokens=False)["input_ids"]
        if len(ids) + reserved <= max_length:
            return ids, shortened
        shortened = True
        overflow_chars = (len(ids) + reserved - max_length) * 4 + 16
        for key in ("product_description", "product_bullet_point", "product_title"):
            # The prompt itself caps these two fields at 900 characters.
            text = str(fields.get(key) or "")[: 900 if key != "product_title" else None]
            if text:
                fields[key] = text[: max(0, len(text) - overflow_chars)]
                break
        else:
            raise ValueError("Prompt cannot fit max_length even with empty product text")
    raise ValueError("Prompt shortening did not converge")


def qwen_training_example(row: Mapping[str, Any], tokenizer, max_length: int) -> dict[str, Any]:
    target = tokenizer(target_text(row["label"]), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    prompt, shortened = fit_prompt_ids(row, tokenizer, max_length, len(target))
    return {
        "input_ids": prompt + target,
        "labels": [-100] * len(prompt) + target,
        "shortened": shortened,
    }


def length_grouped_batches(lengths: Sequence[int], batch_size: int, mega: int, seed: int) -> list[list[int]]:
    """Shuffle, sort within mega-batches by length, cut into batches, shuffle batch order."""
    order = list(range(len(lengths)))
    rng = random.Random(seed)
    rng.shuffle(order)
    batches: list[list[int]] = []
    for start in range(0, len(order), mega):
        chunk = sorted(order[start : start + mega], key=lambda index: lengths[index])
        batches.extend(chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size))
    rng.shuffle(batches)
    return batches


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


def expected_gain_order(products, retrieval_scores: Mapping[str, Mapping[str, float]], gains: Mapping[str, float]) -> list[str]:
    """``rerank_with_sft``'s formula with expected gain in place of the argmax-label gain.

    score = 0.8 * E[gain]/3 + 0.2 * min-max normalized retrieval score, rounded to 4
    decimals, ties by product ID (as in ``rerank_with_sft``).
    """
    from .retrieval import _normalise_retrieval

    normalized = _normalise_retrieval(retrieval_scores, products)
    scored = [(round(0.8 * gains[p.id] / 3 + 0.2 * normalized[p.id], 4), p.id) for p in products]
    return [pid for _, pid in sorted(scored, key=lambda item: (-item[0], item[1]))]
