#!/bin/bash
# Optimized Startup Script for RAG Pipeline Application
# This script:
# 1. Checks for Ollama availability before starting the application
# 2. Verifies required model (llama3:8b) is available
# 3. Starts the Streamlit application with optimized settings

# Set strict error handling - exit if any command fails
set -e

echo "==== RAG Pipeline Startup [Optimized] ===="

# Define Ollama host with fallback to localhost
# Using host.docker.internal allows connecting to host machine from container
OLLAMA_HOST="${OLLAMA_HOST:-host.docker.internal}"
echo "Using Ollama at: $OLLAMA_HOST"

# Create required directories with optimized permissions
mkdir -p /app/data/faiss /app/documents

# Check if Ollama is available with optimized timeout
# Uses curl with specific timeout to prevent long hangs
echo "Checking Ollama server availability..."
if curl -s --max-time 3 --head "http://$OLLAMA_HOST:11434/api/tags" > /dev/null; then
    echo "✅ Ollama server is running at $OLLAMA_HOST:11434"
else
    echo "⚠️ WARNING: Ollama server not available at $OLLAMA_HOST:11434"
    echo "Make sure Ollama is running on your host machine and accessible from the container."
    echo "You can still use the application for document processing, but LLM queries will fail."
fi

# Check if required model exists in Ollama with optimized handling
echo "Checking for required model (llama3:8b)..."
if curl -s --max-time 3 "http://$OLLAMA_HOST:11434/api/tags" | grep -q "llama3:8b"; then
    echo "✅ llama3:8b model found"
else
    echo "⚠️ WARNING: llama3:8b model not found in Ollama"
    echo "Please run 'ollama pull llama3:8b' on your host machine to download the model."
    echo "You can still use the application for document processing, but LLM queries will fail."
fi

# Start Streamlit with optimized settings
echo "Starting Streamlit application..."
# Optimized flags:
# - server.port=8501: Standard port for Streamlit
# - server.address=0.0.0.0: Allow external connections
# - browser.serverAddress=localhost: Proper addressing for UI
# - server.enableCORS=false: Performance improvement for internal use
cd /app && streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --browser.serverAddress=localhost \
    --server.enableCORS=false \
    --server.maxUploadSize=100