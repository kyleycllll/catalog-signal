import argparse
from product_discovery.data_pipeline import import_abo

parser = argparse.ArgumentParser(description="Normalize a local official ABO listings JSONL/GZ file.")
parser.add_argument("--listings", required=True)
parser.add_argument("--output", default="data/processed/products.json")
parser.add_argument("--limit", type=int, default=1000)
args = parser.parse_args()
print(import_abo(args.listings, args.output, args.limit))
