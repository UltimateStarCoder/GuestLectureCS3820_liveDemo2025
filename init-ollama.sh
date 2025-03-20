#!/bin/sh
# Initialization script for Ollama models

# Wait for Ollama service to start up completely
# This prevents connection errors when trying to download the model
echo "Waiting for Ollama service to initialize..."
sleep 5

# Try to install curl if not available
if ! command -v curl &> /dev/null; then
    echo "curl not found, attempting to install..."
    apt-get update && apt-get install -y curl
fi

# Pull Llama 3 model for improved reasoning
echo "Starting Llama 3 model download (this may take several minutes)..."
echo "The application will be ready when you see 'Llama 3 model download complete' message."

# Pull the model with progress output
curl -X POST http://ollama:11434/api/pull -d '{"name":"llama3"}' | tee /tmp/download.log

echo "Llama 3 model download complete! The RAG application is now ready to use."
echo "===================================================================" 