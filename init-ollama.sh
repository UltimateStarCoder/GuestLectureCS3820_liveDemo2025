#!/bin/sh
# Initialization script for Ollama models

# Wait for Ollama service to start up completely
# This prevents connection errors when trying to download the model
sleep 5

# Try to install curl if not available
if ! command -v curl &> /dev/null; then
    echo "curl not found, attempting to install..."
    apt-get update && apt-get install -y curl
fi

# Pull Llama 3 model for improved reasoning
echo "Downloading Llama 3 model for improved RAG performance..."
curl -X POST http://ollama:11434/api/pull -d '{"name":"llama3"}' 