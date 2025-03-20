# Use Python 3.11 slim image as base
# Slim version reduces image size while providing necessary Python functionality
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for document processing
# These are needed for the unstructured package
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt file to container
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size by not caching package downloads
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code to container
COPY app.py .
COPY startup.sh .

# Make startup script executable
RUN chmod +x /app/startup.sh

# Expose port 8501 for Streamlit web interface
EXPOSE 8501

# Use the startup script as entrypoint
CMD ["/app/startup.sh"] 