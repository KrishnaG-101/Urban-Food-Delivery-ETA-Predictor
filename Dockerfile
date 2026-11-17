# Production Dockerfile for Urban Food Delivery ETA Predictor
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specifications
COPY pyproject.toml requirements.txt ./

# Install python dependencies using uv
RUN uv pip install --no-cache --system -r requirements.txt

# Copy application source code
COPY . .

# Expose port for FastAPI server
EXPOSE 9696

# Default command to run FastAPI app via Uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]

