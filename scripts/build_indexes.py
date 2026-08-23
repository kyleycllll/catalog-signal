import argparse
from product_discovery.data_pipeline import validate_catalog
from product_discovery.embeddings import DEFAULT_EMBEDDING_MODEL, FaissTextEmbeddingIndex, build_image_index

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", default="data/processed/esci_products.json")
parser.add_argument("--text-output", default="data/indexes/text_embeddings.faiss")
parser.add_argument("--image-output", default="data/indexes/image_embeddings.npz")
parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
parser.add_argument("--images", action="store_true")
args = parser.parse_args(); products = validate_catalog(args.catalog)
FaissTextEmbeddingIndex.build(products, model_name=args.embedding_model).save(args.text_output)
if args.images: build_image_index(products, args.image_output)
print(f"Indexed {len(products)} products with FAISS at {args.text_output}")
