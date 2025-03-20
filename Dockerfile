## Optimized Multi-Stage Dockerfile for RAG Pipeline Application
# This Dockerfile implements a three-stage build to optimize build times and reduce image size:
# 1. Base stage: System dependencies only
# 2. Dependencies stage: Python packages only
# 3. Final stage: Application code only

# -- BASE STAGE --
# This stage contains only the essential system packages needed for the application
FROM python:3.11-slim AS base

# Install only the necessary system dependencies
# - libmagic1: For file type detection
# - poppler-utils: For PDF processing
# - tesseract-ocr: For OCR capabilities
# - curl: For health checks and API connectivity
# - texlive-base: Basic LaTeX installation
# - texlive-latex-recommended: Standard LaTeX packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    curl \
    texlive-base \
    texlive-latex-recommended \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -- DEPENDENCIES STAGE --
# This stage installs all Python dependencies
# Separating this step optimizes build cache - dependencies only rebuild when requirements.txt changes
FROM base AS dependencies

WORKDIR /deps

# Copy only requirements file first for better layer caching
COPY requirements.txt .

# Install Python dependencies with optimized flags
# - --no-cache-dir: Reduces image size by not caching pip packages
# - --trusted-host: Ensures reliable installations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# -- FINAL STAGE --
# This stage contains only what's necessary to run the application
FROM base AS final

WORKDIR /app

# Create necessary directories in a single RUN command to reduce layers
RUN mkdir -p /app/data/faiss /app/documents

# Copy only the application code
COPY app.py startup.sh ./

# Ensure the startup script is executable
RUN chmod +x /app/startup.sh

# Copy installed dependencies from the dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Set environment variable for health checks
ENV PYTHONUNBUFFERED=1

# Define the entrypoint for the container
# Using the startup script allows for runtime checks and configuration
ENTRYPOINT ["/app/startup.sh"]