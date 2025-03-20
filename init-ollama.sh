#!/bin/sh
# Minimal initialization script for Ollama models

# Wait for Ollama service to be ready
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s --head http://ollama:11434/api/tags > /dev/null; then
        break
    fi
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
done

# Download the model if Ollama is ready
if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
    echo "Downloading Llama 3 model..."
    curl -X POST http://ollama:11434/api/pull -d '{"name":"llama3"}'
else
    echo "ERROR: Ollama service not available after multiple attempts"
    exit 1
fi 