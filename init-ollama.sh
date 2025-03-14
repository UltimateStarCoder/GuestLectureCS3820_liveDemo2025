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

# Pull a lightweight model optimized for edge devices
# TinyLlama 1.1B is much faster than Phi and works well for RAG on limited hardware
echo "Downloading lightweight TinyLlama model for edge devices..."
curl -X POST http://ollama:11434/api/pull -d '{"name":"tinyllama"}' 