FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

# system deps for pandas/numpy (already light)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# ✅ Pre-download VADER into the image to avoid race conditions at runtime
RUN python - <<'PY'
import nltk, os
target = os.environ.get("NLTK_DATA", "/usr/local/share/nltk_data")
os.makedirs(target, exist_ok=True)
nltk.download("vader_lexicon", download_dir=target)
print("Downloaded VADER to:", target)
PY

# (optional) if you prefer to bake your CSV into the image, uncomment the next two lines:
# COPY customer_intelligence_dataset.csv /data/customer_intelligence_dataset.csv
# ENV DATA_LOCAL_PATH=/data/customer_intelligence_dataset.csv

# Copy application code and minimal assets required for runtime
COPY app.py ./app.py
COPY models ./models
COPY data ./data

# Expose & default port (Cloud Run will set PORT automatically)
ENV PORT=8080

EXPOSE 8080

# ✅ Use ONE worker to keep startup simpler & avoid concurrency issues
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT} app:app"]
