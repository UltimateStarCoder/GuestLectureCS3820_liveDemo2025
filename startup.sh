#!/bin/bash
# Startup script for RAG application

mkdir -p /app/data/faiss
mkdir -p /app/documents

echo "==== Checking Ollama Prerequisites ===="
MODEL_NAME="llama3:8b"
# Get OLLAMA_HOST from environment variable, defaulting to host.docker.internal
OLLAMA_HOST="${OLLAMA_HOST:-host.docker.internal}"

echo "Using Ollama host: $OLLAMA_HOST"

# Simple check for Ollama server
curl -s --max-time 5 --head "http://$OLLAMA_HOST:11434/api/tags" > /dev/null
if [ $? -eq 0 ]; then
  echo "✅ Ollama server is running!"
  
  # Check if the model exists
  if curl -s --max-time 5 "http://$OLLAMA_HOST:11434/api/show" -d "{\"name\":\"$MODEL_NAME\"}" | grep -q "model"; then
    echo "✅ $MODEL_NAME model is available!"
  else
    echo "⚠️ $MODEL_NAME model not found on Ollama server."
    echo "   Please run 'ollama pull $MODEL_NAME' on your host machine."
  fi
else
  echo "❌ OLLAMA SERVER NOT RUNNING - Please start Ollama on your host"
fi

echo "==== Starting RAG Pipeline ===="
echo "Access at: http://localhost:8501"

# Start Streamlit
exec python -m streamlit run app.py --server.address=0.0.0.0 --server.headless=true 