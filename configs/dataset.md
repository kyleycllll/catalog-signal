# Dataset contract

Source: Amazon Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search, from `amazon-science/esci-data` under Apache-2.0.

Join examples and products on `product_locale` plus `product_id`. The project uses US English rows from `small_version` for a Colab-sized experiment. Official test queries remain test-only; validation query IDs are deterministically separated from official training rows.

Required example fields: `example_id`, `query_id`, `query`, `product_id`, `product_locale`, `esci_label`, `small_version`, and `split`.

Required product fields: `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, and `product_locale`.

Labels are E, S, C, and I. The preparation script balances its limited training/evaluation samples by label and writes exact counts to `data/esci/manifest.json`.

The source does not provide price. Any future price enrichment must record source, timestamp, currency, join coverage, and missingness; missing prices must never be synthesized.
