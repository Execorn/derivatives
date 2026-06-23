# Production Dockerfile for GPU-accelerated option pricing and deep hedging
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install remaining python packages
COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torchsde iisignature

# Copy source and artifact files
COPY src/ /app/src/
COPY data/ /app/data/
COPY artifacts/ /app/artifacts/

# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Start the uvicorn API server by default
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
