#!/bin/zsh
set -e
cd /Users/kyle/Developer/Personal/finetuning
export PYTHONPATH=src
PY=/private/tmp/catalog-signal-eval-venv/bin/python
for V in A_balanced_random B_hard_negative; do
  echo "=== train $V $(date)"
  $PY scripts/train_esci_reranker.py --variant $V --output-dir models/ablation/qwen_$V
done
for V in A_balanced_random B_hard_negative; do
  for C in models/ablation/qwen_$V/checkpoint-*; do
    echo "=== validate $C $(date)"
    $PY scripts/evaluate_reranker_validation.py --adapter $C
  done
done
echo "=== validate historical $(date)"
$PY scripts/evaluate_reranker_validation.py --adapter models/esci-qwen-lora --output-dir models/ablation/historical_validation
echo "=== CHAIN DONE $(date)"
