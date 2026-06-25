# Stage 1: Build & Compile CUDA extensions and python packages
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and filter out torch packages since they are pre-installed in base image
COPY src/requirements.txt /build/src/requirements.txt
RUN grep -v -E "torch|torchaudio|torchvision" /build/src/requirements.txt > /build/src/requirements_filtered.txt && \
    pip install --user --no-cache-dir -r /build/src/requirements_filtered.txt && \
    pip install --user --no-cache-dir torchsde iisignature cachetools confluent-kafka redis

# Copy setup.py and source to compile the CUDA extension
COPY setup.py /build/setup.py
COPY src/ /build/src/

# Compile the CUDA extension
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN python setup.py build_ext --inplace

# Stage 2: Final lightweight runtime environment
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH="/app/src:${PYTHONPATH}"

WORKDIR /app

# Copy python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy source and compiled library files from builder
COPY --from=builder /build/src/ /app/src/

# Copy artifacts (weights and normalizers)
COPY artifacts/ /app/artifacts/

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Start the uvicorn API server by default
CMD ["uvicorn", "deepvol.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
