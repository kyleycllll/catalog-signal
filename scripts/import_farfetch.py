import argparse
from product_discovery.data_pipeline import import_farfetch

parser = argparse.ArgumentParser(description="Normalize the downloaded Kaggle Farfetch dataset.")
parser.add_argument("--metadata", default="data/raw/farfetch/images/farfetch_fashion_dataset_images_crawlfeeds.json")
parser.add_argument("--source-root", default="data/raw/farfetch")
parser.add_argument("--output", default="data/processed/products.json")
args = parser.parse_args(); print(import_farfetch(args.metadata, args.output, args.source_root))
