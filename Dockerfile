# Multi-stage Dockerfile for cryptofeed
# Production-ready container image optimized for size, security, and performance

# =============================================================================
# Stage 1: Builder - Compile dependencies and build wheels
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

LABEL stage=builder

WORKDIR /build

# Install build dependencies for compiling Python extensions
# - gcc, g++, build-essential: C/C++ compilers for Cython and native extensions
# - librdkafka-dev: Development files for Kafka client library
# - git: Required for some pip dependencies that install from git repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    librdkafka-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .
COPY README.md INSTALL.md CHANGES.md .

# Create wheels directory
RUN mkdir -p /wheels

# Install Cython first (required for setup.py)
RUN pip install --no-cache-dir Cython

# Build wheels for all dependencies
# --wheel-dir: Output directory for compiled wheels
# --find-links: Use local wheels if available
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Install aiokafka for Kafka backend support
RUN pip wheel --no-cache-dir --wheel-dir /wheels aiokafka>=0.7.0

# Install protobuf for message serialization
RUN pip wheel --no-cache-dir --wheel-dir /wheels protobuf>=4.21.0

# Install prometheus_client for metrics exposure
RUN pip wheel --no-cache-dir --wheel-dir /wheels prometheus_client>=0.17.0

# Copy source code for building cryptofeed wheel
COPY cryptofeed/ /build/cryptofeed/

# Build cryptofeed wheel (includes Cython extension compilation)
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Build-time arguments for image metadata
ARG VERSION=dev
ARG BUILD_TIMESTAMP
ARG GIT_COMMIT_SHA
ARG GIT_BRANCH

# Image metadata labels for traceability and versioning
# These labels follow OCI image spec and Docker best practices
LABEL maintainer="cryptofeed@example.com" \
      description="Cryptofeed - Cryptocurrency market data ingestion platform" \
      version="${VERSION}" \
      build_timestamp="${BUILD_TIMESTAMP}" \
      git_commit_sha="${GIT_COMMIT_SHA}" \
      git_branch="${GIT_BRANCH}" \
      org.opencontainers.image.title="cryptofeed" \
      org.opencontainers.image.description="Cryptocurrency market data ingestion platform" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_TIMESTAMP}" \
      org.opencontainers.image.revision="${GIT_COMMIT_SHA}" \
      org.opencontainers.image.source="https://github.com/bmoscon/cryptofeed"

WORKDIR /app

# Install runtime dependencies only (no build tools)
# - librdkafka1: Runtime library for Kafka client
# - curl: Required for health checks in Docker Compose and k3s
RUN apt-get update && apt-get install -y --no-install-recommends \
    librdkafka1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled wheels from builder stage
COPY --from=builder /wheels /wheels

# Install all packages from wheels (faster than pip install from source)
# This includes cryptofeed with compiled Cython extensions
RUN pip install --no-cache-dir --find-links /wheels /wheels/*.whl \
    && rm -rf /wheels

# Create non-root user 'cryptofeed' with UID 1001
# Security best practice: never run containers as root
RUN useradd -m -u 1001 -s /bin/bash cryptofeed \
    && chown -R cryptofeed:cryptofeed /app

# Create directory for Prometheus multiprocess metrics
# Must be writable by non-root user
RUN mkdir -p /tmp/prometheus \
    && chown -R cryptofeed:cryptofeed /tmp/prometheus

# Switch to non-root user
USER cryptofeed

# Set Python environment variables
# PYTHONUNBUFFERED=1: Disable output buffering for real-time logs
# PYTHONDONTWRITEBYTECODE=1: Disable .pyc file generation (not needed in containers)
# PROMETHEUS_MULTIPROC_DIR: Directory for Prometheus multiprocess mode
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

# Expose ports
# 8080: Health check endpoint
# 9090: Prometheus metrics endpoint
EXPOSE 8080 9090

# Default command: Run cryptofeed.run module with config file
# Users can override CMD to pass different arguments or run different commands
# Example: docker run cryptofeed --config /custom/config.yaml
# Example: docker run cryptofeed python -c "import cryptofeed; print(cryptofeed.__version__)"
CMD ["python", "-m", "cryptofeed.run", "--config", "/config/config.yaml"]
