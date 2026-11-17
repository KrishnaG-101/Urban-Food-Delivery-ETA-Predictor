# Production Dockerfile for Urban Food Delivery ETA Predictor
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# libgomp1 is strictly required by LightGBM OpenMP runtime on Debian/Ubuntu slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, deterministic package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specifications
COPY pyproject.toml requirements.txt ./

# Install python dependencies using uv
RUN uv pip install --no-cache --system -r requirements.txt

# Copy source code and raw data needed for training
COPY data/raw/food_delivery.csv data/raw/food_delivery.csv
COPY predict.py train.py serve.py ./

# Train and bake model artifact directly into the image
RUN python train.py

# Copy remaining project files (tests, configs)
COPY . .

# Expose port for FastAPI service
EXPOSE 9696

# Launch Uvicorn with dynamic cloud PORT variable handling (defaults to 9696)
CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-9696}"]
