#!/bin/zsh
# Fixed-candidate TEST evaluation of the frozen selections (validation_selection.json).
set -e
cd /Users/kyle/Developer/Personal/finetuning
export PYTHONPATH=src
PY=/private/tmp/catalog-signal-eval-venv/bin/python
echo "=== ce $(date)"
$PY scripts/run_reranker_ablation_eval.py --kind cross_encoder --model models/ablation/ce_A_balanced_random/epoch-3 --name cross_encoder
echo "=== qwen A $(date)"
$PY scripts/run_reranker_ablation_eval.py --kind qwen --model models/ablation/qwen_A_balanced_random/checkpoint-625 --name qwen_A_balanced_random
echo "=== qwen B $(date)"
$PY scripts/run_reranker_ablation_eval.py --kind qwen --model models/ablation/qwen_B_hard_negative/checkpoint-625 --name qwen_B_hard_negative
echo "=== historical rerun (latency/determinism check only) $(date)"
$PY scripts/run_reranker_ablation_eval.py --kind qwen --model models/esci-qwen-lora --name historical_qwen_rerun
echo "=== TEST DONE $(date)"
