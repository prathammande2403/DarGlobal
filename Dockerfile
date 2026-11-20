FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin app
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
RUN mkdir -p /app/data/chroma_updated /app/data/raw/cache /home/app/.cache/fastembed \
    && chown -R app:app /app/data /home/app/.cache

USER app

EXPOSE 8000 8501

ENTRYPOINT ["sh", "/app/docker-entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
