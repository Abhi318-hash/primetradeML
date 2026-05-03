# ─────────────────────────────────────────────────────────────
# Dockerfile — MLOps Batch Job
# Primetrade.ai ML/MLOps Engineering Internship Assessment
# ─────────────────────────────────────────────────────────────

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY run.py config.yaml data.csv ./

# Run the pipeline at build time so the image is pre-validated
RUN python run.py \
    --input data.csv \
    --config config.yaml \
    --output metrics.json \
    --log-file run.log

# Print metrics.json to stdout on container start
# Exit code: 0 = success (cat succeeds), non-zero if pipeline failed
CMD ["cat", "metrics.json"]
