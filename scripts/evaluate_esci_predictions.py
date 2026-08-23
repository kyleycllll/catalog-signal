"""Score JSONL rows containing at least {gold, prediction}; never invents benchmark numbers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from product_discovery.evaluation import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/esci_metrics.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = classification_metrics([row["gold"] for row in rows], [row["prediction"] for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
