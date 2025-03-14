# Use Python 3.11 slim image as base
# Slim version reduces image size while providing necessary Python functionality
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for document processing
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    pandoc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file to container
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size by not caching package downloads
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code to container
COPY app.py .

# Expose port 8501 for Streamlit web interface
EXPOSE 8501

# Run the Streamlit application
# --server.address=0.0.0.0 allows connections from any IP
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"] 