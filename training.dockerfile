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

# 4) Create artifact dir (model) – data will be mounted at runtime
RUN mkdir -p /app/models

# 5) Default command runs training
#    Pass --data and --output if your train script supports args;
#    otherwise it can use these env defaults.
ENV DATA_PATH=/app/data/Books_10k.jsonl
ENV OUTPUT_PATH=/app/models/sentiment_pipeline.pkl
# If your train script reads envs, keep as-is; otherwise pass as args below.
CMD ["bash", "-lc", "python code/train_pipeline.py --data $DATA_PATH --output $OUTPUT_PATH"]