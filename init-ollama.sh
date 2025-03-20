#!/bin/sh
# Initialization script for Ollama models

# Wait for Ollama service to start up completely
echo "Waiting for Ollama service to initialize..."

# Instead of a fixed sleep, keep checking until the service is available
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s --head http://ollama:11434/api/tags > /dev/null; then
        echo "Ollama service is ready!"
        break
    fi
    echo "Waiting for Ollama service... ($((RETRY_COUNT+1))/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Timed out waiting for Ollama service to become available"
    echo "Will attempt to proceed anyway..."
fi

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