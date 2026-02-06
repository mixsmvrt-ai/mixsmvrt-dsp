# Dockerfile for MixSmvrt DSP service on Fly.io

FROM python:3.11-slim

# System deps for audio/DSP stack: ffmpeg, libsndfile, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       libsndfile1 \
       libsamplerate0 \
       libfftw3-3 \
       libyaml-0-2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Fly sets $PORT; default to 8080 for local/dev
ENV PORT=8080

EXPOSE 8080

# Use uvicorn to serve FastAPI app
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
