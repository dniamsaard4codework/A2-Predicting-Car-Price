
# Multi-stage build using uv for efficient Python dependency management
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Set environment variables
ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY .python-version .
COPY uv.lock .

# Create virtual environment and install dependencies
RUN uv sync --frozen --no-cache

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

COPY app/app.py ./

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application files
COPY --chown=appuser:appuser app/app.py ./
COPY --chown=appuser:appuser model/car_price.model ./
COPY --chown=appuser:appuser model/best_model.pkl ./

# Copy notebooks and data
# COPY --chown=appuser:appuser notebook/ ./notebook/
# COPY --chown=appuser:appuser data/ ./data/

# Switch to non-root user
USER appuser

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set environment variables
ENV PORT=8050
ENV DEBUG=False
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8050')" || exit 1

# Run the application
CMD ["python", "app.py"]