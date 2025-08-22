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

COPY app.py ./app.py

# Expose & default port
ENV PORT=8501

EXPOSE 8501

# ✅ Use ONE worker to keep startup simpler & avoid concurrency issues
CMD ["gunicorn", "-b", "0.0.0.0:8501", "-w", "1", "app:app"]
