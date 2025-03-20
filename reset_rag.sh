#!/bin/bash
# Optimized Reset Script for RAG Pipeline
# This script:
# 1. Stops all running containers
# 2. Cleans Docker volumes
# 3. Rebuilds the application with optimized settings
# 4. Restarts the application
# 5. Shows clear status for each operation

# Set strict error handling
set -e

echo "=== RAG Pipeline Reset [Optimized] ==="

# Function for formatting console output
format_status() {
  local status=$1
  local message=$2
  
  if [ "$status" = "success" ]; then
    echo -e "✅ $message"
  elif [ "$status" = "info" ]; then
    echo -e "ℹ️ $message"
  elif [ "$status" = "warn" ]; then
    echo -e "⚠️ $message"
  else
    echo -e "❌ $message"
  fi
}

# Stop running containers - forcefully and with proper cleanup
format_status "info" "Stopping running containers..."
docker-compose down --volumes --remove-orphans 2>/dev/null || true
format_status "success" "Containers stopped"

# Clean Docker volumes for a fresh start
# This ensures no stale data remains
format_status "info" "Cleaning Docker volumes..."
docker volume rm -f guestlecturecs3820_livedemo2025_faiss_data guestlecturecs3820_livedemo2025_doc_storage 2>/dev/null || true
format_status "success" "Docker volumes cleaned"

# Verify Ollama is running - essential for RAG functionality
format_status "info" "Checking Ollama availability..."
OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
if curl -s --max-time 3 --head "http://$OLLAMA_HOST:11434/api/tags" > /dev/null; then
  format_status "success" "Ollama is running"
else
  format_status "warn" "Ollama not available - ensure it's running on your host machine"
  format_status "warn" "Application will still build, but LLM features require Ollama"
fi

# Use optimized build script
format_status "info" "Building with optimized settings..."
if [ -f "./build-fast.sh" ]; then
  chmod +x ./build-fast.sh
  ./build-fast.sh  # Using our optimized build script
  format_status "success" "Build completed using optimized build script"
else
  format_status "info" "Using standard build process (optimized script not found)"
  # Fallback to regular build with BuildKit enabled
  export DOCKER_BUILDKIT=1
  export COMPOSE_DOCKER_CLI_BUILD=1
  docker-compose build
  format_status "success" "Build completed using standard process"
fi

# Start the application
format_status "info" "Starting RAG pipeline application..."
docker-compose up -d
format_status "success" "RAG pipeline started"

# Display access information
format_status "info" "RAG Application is now accessible at: http://localhost:8501"
echo ""
format_status "info" "Waiting for application to be ready..."

# More reliable application readiness check
max_attempts=10
attempt=0
while [ $attempt -lt $max_attempts ]; do
  if curl -s --max-time 3 --head "http://localhost:8501" > /dev/null; then
    format_status "success" "Application is ready!"
    format_status "info" "Open your browser at: http://localhost:8501"
    break
  fi
  
  attempt=$((attempt+1))
  echo -n "."
  sleep 2
done

if [ $attempt -eq $max_attempts ]; then
  format_status "warn" "Application not responding yet, but it may still be starting up"
  format_status "info" "Try accessing http://localhost:8501 in your browser"
fi