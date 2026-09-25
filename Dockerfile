FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY models/ablation/ce_A_balanced_random/epoch-3 ./models/ablation/ce_A_balanced_random/epoch-3
RUN pip install --no-cache-dir ".[serving]"
EXPOSE 8000
CMD ["uvicorn", "product_discovery.api:app", "--host", "0.0.0.0", "--port", "8000"]
