# ===== Dockerfile.train =====
FROM python:3.10-slim

# 1) System prep
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 2) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy code (training script + custom transformer)
COPY code ./code
COPY data ./data

# Make sure Python can import from /app/code
ENV PYTHONPATH=/app/code

# 4) Artifacts/output directory (mounted in run is fine; this just ensures it exists)
RUN mkdir -p /app/output

# 5) Defaults (override via docker run -e … or CLI args)
ENV TRAIN_CSV=/app/data/Books_10k.jsonl \
    CONFIG_PATH=/app/code/sentimentpredictor/config \
    OUTDIR=/app/output


# If your train script reads envs, keep as-is; otherwise pass as args below.
CMD ["bash", "-lc", "python code/sentimentpredictor/train/training_pipeline.py --train-file \"$TRAIN_CSV\" --config-path \"$CONFIG_PATH\" --outdir \"$OUTDIR\""]