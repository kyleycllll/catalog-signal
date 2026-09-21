"""Local, in-process inference for the saved ESCI QLoRA adapter.

This reproduces the Colab notebook's pointwise generative classifier exactly at
the prompt/decoding level: the same raw (non-chat-template) prompt, greedy
decoding, the same ``"label":"X"`` parser, and the same fallback to ``I`` when no
label is parsed. It is *not* a pairwise, listwise, or cross-encoder ranker.

The adapter was trained against a 4-bit NF4 (bitsandbytes, double-quantized)
base. bitsandbytes does not run on Apple silicon, so ``base_weights="nf4"``
*emulates* that quantization: each base ``nn.Linear`` weight is quantized to the
16 NF4 levels in blocks of 64 with double-quantized (8-bit dynamic, blocks of
256) absmax scales, then dequantized for fp16 compute. The embedding/lm_head are
not quantized, as in bitsandbytes. Agreement with the saved Colab predictions
is measured, not assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import ModelHealth, ModelPrediction, Product, RelevanceLabel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LABEL_HELP = "E=exact match, S=usable substitute, C=complement/accessory, I=irrelevant"
LABEL_PATTERN = re.compile(r'"label"\s*:\s*"([ESCI])"')


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text == "nan" else text


def classification_prompt(row: Mapping[str, Any]) -> str:
    """Byte-for-byte copy of the notebook's ``classification_prompt``."""
    return f'''Classify the relationship between a shopping query and product.
{LABEL_HELP}.
Return only JSON: {{"label":"E|S|C|I"}}
Query: {_clean(row['query'])}
Product title: {_clean(row['product_title'])}
Brand: {_clean(row['product_brand'])}
Color: {_clean(row['product_color'])}
Bullet points: {_clean(row['product_bullet_point'])[:900]}
Description: {_clean(row['product_description'])[:900]}
JSON:'''.strip()


def prompt_row(query: str, product: Product) -> dict[str, Any]:
    """Map a Product to the notebook serving cell's row fields."""
    return {
        "query": query,
        "product_title": product.title,
        "product_brand": product.brand or "",
        "product_color": product.colour or "",
        "product_bullet_point": product.bullet_points or "",
        "product_description": product.description,
    }


def parse_label(text: str) -> tuple[str, bool]:
    """Return the parsed label and whether parsing succeeded (fallback is ``I``)."""
    match = LABEL_PATTERN.search(text)
    return (match.group(1), True) if match else ("I", False)


NF4_LEVELS = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


def dynamic_map(signed: bool = True, max_exponent_bits: int = 7, total_bits: int = 8):
    """bitsandbytes ``create_dynamic_map`` (the 8-bit code used for double quantization)."""
    import torch

    data: list[float] = []
    non_sign_bits = total_bits - 1
    for i in range(max_exponent_bits):
        items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1 if signed
                    else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1)
        boundaries = torch.linspace(0.1, 1, items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
    data += [0.0, 1.0]
    data += [0.0] * (2**total_bits - len(data))
    return torch.tensor(sorted(data))


def _nearest(values, code):
    import torch

    index = torch.bucketize(values, (code[1:] + code[:-1]) / 2)
    return code[index]


def nf4_roundtrip(weight, blocksize: int = 64, double_quant: bool = True):
    """Quantize a weight to blockwise NF4 and dequantize it (emulated bitsandbytes)."""
    import torch

    flat = weight.detach().float().reshape(-1)
    pad = (-flat.numel()) % blocksize
    blocks = torch.nn.functional.pad(flat, (0, pad)).reshape(-1, blocksize)
    absmax = blocks.abs().amax(dim=1)
    safe = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    levels = torch.tensor(NF4_LEVELS)
    quantized = _nearest((blocks / safe[:, None]).clamp(-1, 1), levels)
    if double_quant:
        offset = absmax.mean()
        centred = absmax - offset
        pad2 = (-centred.numel()) % 256
        groups = torch.nn.functional.pad(centred, (0, pad2)).reshape(-1, 256)
        scale = groups.abs().amax(dim=1).clamp_min(1e-12)
        code = dynamic_map()
        absmax = (_nearest(groups / scale[:, None], code) * scale[:, None]).reshape(-1)[: absmax.numel()] + offset
    restored = (quantized * absmax[:, None]).reshape(-1)[: flat.numel()]
    return restored.reshape(weight.shape)


def emulate_nf4_(model, blocksize: int = 64, double_quant: bool = True) -> int:
    """Replace every base Linear weight (not lm_head) with its NF4 round trip."""
    import torch

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
            with torch.no_grad():
                module.weight.copy_(nf4_roundtrip(module.weight, blocksize, double_quant).to(module.weight.dtype))
            count += 1
    return count


@dataclass
class GenerationResult:
    label: str
    parsed: bool
    raw: str


class LocalAdapterReranker:
    """Batched greedy generation with the LoRA adapter (or the frozen base model)."""

    def __init__(
        self,
        adapter_dir: str | None,
        base_model: str = BASE_MODEL,
        device: str | None = None,
        dtype: str = "bfloat16",
        batch_size: int = 8,
        max_length: int = 1536,
        max_new_tokens: int = 16,
        base_weights: str = "full",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype_name = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.base_weights = base_weights
        if base_weights == "nf4":
            model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
            self.quantized_linear_layers = emulate_nf4_(model)
            model = model.to(getattr(torch, dtype))
        elif base_weights == "full":
            model = AutoModelForCausalLM.from_pretrained(base_model, dtype=getattr(torch, dtype))
        else:
            raise ValueError("base_weights must be 'full' or 'nf4'")
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        self.model = model.to(self.device).eval()

    def generate(self, prompts: list[str]) -> list[GenerationResult]:
        torch = self.torch
        results: list[GenerationResult] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output[:, inputs["input_ids"].shape[1] :]
            for text in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
                label, parsed = parse_label(text)
                results.append(GenerationResult(label=label, parsed=parsed, raw=text))
        return results

    # ModelClient-compatible surface used by retrieval.rerank_with_sft.
    async def health(self) -> ModelHealth:
        return ModelHealth(
            status="ok",
            model_kind="fine_tuned_adapter",
            adapter_loaded=bool(self.adapter_dir),
            model_version=f"local:{self.adapter_dir}:{self.base_weights}:{self.dtype_name}",
            base_model=self.base_model,
        )

    async def rerank(self, query: str, products: list[Product]) -> list[ModelPrediction]:
        generations = self.generate([classification_prompt(prompt_row(query, p)) for p in products])
        return [
            ModelPrediction(
                product_id=product.id,
                label=RelevanceLabel(result.label),
                rationale=f"Adapter ESCI prediction: {result.label}",
            )
            for product, result in zip(products, generations)
        ]
