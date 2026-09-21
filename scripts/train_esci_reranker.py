"""Config-driven QLoRA training of the Qwen ESCI classifier (pointwise, generative).

Reproducible local replacement for the Colab notebook's training cell. The base is
Qwen2.5-0.5B-Instruct with an emulated 4-bit NF4 double-quantized base (as used for
the reproduced historical adapter), frozen in fp16; only LoRA weights (fp32) train.
Loss is on the ``{"label":"X"}`` target + EOS only, and the label is never truncated.

Per run (``--output-dir``) this writes run_config.json, data_manifest.json,
training_metrics.json, adapter checkpoints (adapter_config.json +
adapter_model.safetensors) and artifact hashes. Validation predictions are produced
by ``scripts/evaluate_reranker_validation.py``. Official test data is never read.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from product_discovery.local_reranker import emulate_nf4_
from product_discovery.provenance import run_provenance, sha256_file
from product_discovery.reranker_training import length_grouped_batches, qwen_training_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_reranker_ablation.json")
    parser.add_argument("--variant", required=True, help="Training variant name in training_data_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke runs only")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    qwen = config["qwen"]
    run_dir = Path(config["output_root"]) / config["run_id"]
    data_manifest = json.loads((run_dir / "training_data_manifest.json").read_text(encoding="utf-8"))
    variant = data_manifest["variants"][args.variant]
    data_path = Path(variant["path"])
    if sha256_file(data_path) != variant["sha256"]:
        raise SystemExit("Training data changed since the manifest was written")
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").split("\n") if line]
    base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
    import pyarrow.parquet as pq

    held_out = set(pq.read_table(base["labels"], columns=["query_id", "prepared_split"],
                                 filters=[("prepared_split", "in", ["validation", "test"])])
                   .column("query_id").to_pylist())
    if any(row["query_id"] in held_out for row in rows):
        raise SystemExit("Training rows contain a validation/test query")

    seed = config["seed"]
    torch.manual_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(qwen["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    examples = [qwen_training_example(row, tokenizer, qwen["max_length"]) for row in rows]
    lengths = [len(example["input_ids"]) for example in examples]
    target_lengths = [sum(label != -100 for label in example["labels"]) for example in examples]

    model = AutoModelForCausalLM.from_pretrained(qwen["base_model"], dtype=torch.float32)
    quantized_layers = emulate_nf4_(model)
    model = model.to(torch.float16)
    model.config.use_cache = False
    lora_cfg = qwen["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"], task_type="CAUSAL_LM", target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if qwen.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch_size, accumulation = qwen["per_device_batch_size"], qwen["gradient_accumulation"]
    batches: list[list[int]] = []
    for epoch in range(qwen["epochs"]):
        batches.extend(length_grouped_batches(lengths, batch_size, 64, seed + epoch))
    total_steps = math.ceil(len(batches) / accumulation)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
        batches = batches[: total_steps * accumulation]
    checkpoints = sorted({max(1, round(total_steps * f)) for f in qwen["checkpoint_fractions"]})

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=qwen["learning_rate"], weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps * qwen["warmup_ratio"]), total_steps)
    # fp16 activations need loss scaling (as fp16 AMP did on Colab).
    scaler = torch.amp.GradScaler(device, init_scale=2.0**12)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "variant": args.variant, "config": args.config, "config_sha256": sha256_file(args.config),
        "qwen": qwen, "seed": seed, "device": device, "trainable_parameters": trainable,
        "quantized_linear_layers": quantized_layers, "micro_batches": len(batches),
        "optimizer_steps": total_steps, "checkpoint_steps": checkpoints,
        "effective_batch_size": batch_size * accumulation, "smoke_max_steps": args.max_steps,
        "provenance": run_provenance(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.output_dir / "data_manifest.json").write_text(json.dumps({
        "variant": args.variant, **variant,
        "rows_prompt_shortened_to_fit": sum(example["shortened"] for example in examples),
        "token_length": {"mean": sum(lengths) / len(lengths), "max": max(lengths)},
        "leakage_check": "no training query is a validation or test query (asserted)",
        "training_data_manifest": str(run_dir / "training_data_manifest.json"),
        "training_data_manifest_sha256": sha256_file(run_dir / "training_data_manifest.json"),
    }, indent=2), encoding="utf-8")

    log: list[dict] = []
    saved: dict[str, dict] = {}
    model.train()
    started = time.perf_counter()
    running, running_tokens, step = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for micro, batch in enumerate(batches, start=1):
        # Left padding with explicit position IDs, as in batched generate(), so every
        # target sits in the last ``keep`` positions and only those are projected to
        # the vocabulary (full-sequence logits would be ~2.5 GB per micro-batch).
        width = max(lengths[i] for i in batch)
        keep = max(target_lengths[i] for i in batch) + 1
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for j, index in enumerate(batch):
            ids, target = examples[index]["input_ids"], examples[index]["labels"]
            input_ids[j, width - len(ids):] = torch.tensor(ids)
            labels[j, width - len(ids):] = torch.tensor(target)
            attention[j, width - len(ids):] = 1
        positions = (attention.cumsum(-1) - 1).clamp(min=0)
        output = model(input_ids=input_ids.to(device), attention_mask=attention.to(device),
                       position_ids=positions.to(device), logits_to_keep=keep)
        logits = output.logits[:, :-1, :].float()
        shifted = labels[:, width - keep + 1:].to(device)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), shifted.reshape(-1), ignore_index=-100)
        scaler.scale(loss / accumulation).backward()
        running += float(loss.detach())
        running_tokens += 1
        if micro % accumulation == 0:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, qwen["max_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == total_steps:
                entry = {"step": step, "loss": running / running_tokens, "grad_norm": grad_norm,
                         "lr": scheduler.get_last_lr()[0], "loss_scale": scaler.get_scale(),
                         "elapsed_s": round(time.perf_counter() - started, 1)}
                log.append(entry)
                print(json.dumps(entry), flush=True)
                running, running_tokens = 0.0, 0
            if step in checkpoints:
                target = args.output_dir / f"checkpoint-{step}"
                model.save_pretrained(target)
                tokenizer.save_pretrained(target)
                saved[f"checkpoint-{step}"] = {
                    "path": str(target), "step": step,
                    "adapter_sha256": sha256_file(target / "adapter_model.safetensors"),
                    "adapter_config_sha256": sha256_file(target / "adapter_config.json"),
                }
            if step >= total_steps:
                break

    seconds = time.perf_counter() - started
    metrics = {
        "variant": args.variant, "optimizer_steps": step, "examples_seen": sum(len(b) for b in batches),
        "training_seconds": round(seconds, 1), "examples_per_second": round(sum(len(b) for b in batches) / seconds, 3),
        "loss_log": log, "checkpoints": saved, "provenance": run_provenance(),
    }
    (args.output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "loss_log"}, indent=2))


if __name__ == "__main__":
    main()
