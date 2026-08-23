# Colab checklist

Use `notebooks/esci_sft_reranker_colab.ipynb`; the old `train_lora_colab.ipynb` belongs to the superseded toy intent experiment.

1. Upload the new notebook to Colab and select a T4 GPU.
2. Run the installation/data cells.
3. For the hard-negative ablation, first run `scripts/prepare_hard_negatives.py`, upload the resulting `baseline.jsonl` or `hard_negative.jsonl` plus `manifest.json` to `/content/esci-training-variants`, and set `TRAINING_VARIANT` in the data cell. Run one Colab job per variant.
4. Run the frozen-base evaluation cell. This records the **before** result.
5. Run the QLoRA training/evaluation cell. This records the **after** result, training time, selected variant metadata, and learned attention-adapter matrices.
6. Mount Drive and save the adapter plus comparison artifacts.
7. Add private `NGROK_AUTHTOKEN` and `MODEL_API_KEY` Colab secrets.
8. Run the serving cell and copy its URL into local `.env` as `FINETUNED_MODEL_URL`.
9. Put the same secret value in local `.env` as `FINETUNED_MODEL_API_KEY`.
10. Keep Colab running while using localhost. When Colab stops, localhost correctly returns 503 because the required fine-tuned model is no longer available.
