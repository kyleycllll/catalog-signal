import argparse
from product_discovery.training_data import prepare_file

parser = argparse.ArgumentParser(description="Create leakage-safe LoRA train/validation/test JSONL splits.")
parser.add_argument("--source", required=True)
parser.add_argument("--output-dir", default="data/intent_splits")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args(); print(prepare_file(args.source, args.output_dir, args.seed))
