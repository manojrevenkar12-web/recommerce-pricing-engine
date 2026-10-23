FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RPE_ROOT=/app \
    RPE_DATA=/data \
    RPE_MODELS=/models \
    RPE_REPORTS=/reports

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY config ./config
COPY scenarios ./scenarios

RUN useradd --uid 10001 --create-home rpe \
 && mkdir -p /data /models /reports \
 && chown rpe /data /models /reports
USER rpe

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s CMD curl -fsS localhost:8000/health || exit 1
CMD ["rpe", "serve", "--port", "8000"]
