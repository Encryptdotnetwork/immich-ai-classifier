# Immich AI Classifier — foundation image.
# Single-purpose Python container. Reads asset files off a read-only bind-mount
# of the Immich library; writes results back via the Immich API (later tasks).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ffmpeg is required by MoviePy for video frame extraction (signals.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Dependencies next for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY app/ ./app/

# Category taxonomy (config/categories.yaml). Shipped as a sensible default and
# loaded at startup; docker-compose bind-mounts ./config over this so it stays
# user-editable without rebuilding the image.
COPY config/ ./config/

# Default command runs the verify / acceptance test. Pass an asset id via the
# ASSET_ID env var, or override the command:
#   docker compose run --rm immich-ai-classifier python -m app.main <asset_id>
CMD ["python", "-m", "app.main"]
