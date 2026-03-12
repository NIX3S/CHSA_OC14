FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_PATH=/app/model \
    MAX_TOKENS=512 \
    TEMPERATURE=0.1 \
    PORT=8000

# Fix Python + deps système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Renomme ton fichier pour Docker
COPY week4_api_fastapi.py app.py
COPY week4_evaluation.py .

RUN mkdir -p logs && chmod 777 logs
RUN useradd -m -u 1000 chsa && chown -R chsa:chsa /app
USER chsa

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info"]
