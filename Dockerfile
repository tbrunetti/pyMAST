# pyMAST Docker image
# Usage:
#   docker build -t pymast:latest .
#   docker run --rm pymast:latest pytest tests/ -v
#   docker run --rm -p 8888:8888 pymast:latest jupyter lab --ip=0.0.0.0 --no-browser

FROM python:3.14-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY pymast/ ./pymast/

# Install package + dev/jupyter extras
RUN uv pip install --system ".[dev,jupyter]"

# Copy tests and examples (after install to leverage cache)
COPY tests/ ./tests/
COPY examples/ ./examples/

# Default: run test suite
EXPOSE 8888
CMD ["pytest", "tests/", "-v", "--tb=short"]
