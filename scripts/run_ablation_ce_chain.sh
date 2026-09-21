#!/bin/zsh
set -e
cd /Users/kyle/Developer/Personal/finetuning
export PYTHONPATH=src
for V in A_balanced_random B_hard_negative; do
  echo "=== ce $V $(date)"
  /private/tmp/catalog-signal-eval-venv/bin/python scripts/train_esci_cross_encoder.py --variant $V --output-dir models/ablation/ce_$V
done
echo "=== CE DONE $(date)"
