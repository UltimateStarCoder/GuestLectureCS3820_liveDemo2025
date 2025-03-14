#!/bin/bash

echo "🚀 Starting Simple RAG Pipeline with Universal Document Support"
echo "This script will attempt to launch the application with GPU support if available."
echo "If you don't have GPU support, it will fall back to CPU mode."

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed or not in PATH. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA drivers are available for Docker
if docker info | grep -q "Runtimes:.*nvidia" || docker info | grep -q "Default Runtime:.*nvidia"; then
    echo "✅ NVIDIA GPU support detected! Using GPU for accelerated processing."
    echo "Starting with GPU support..."
    docker-compose up -d
else
    echo "⚠️ No NVIDIA GPU support detected. Starting in CPU-only mode."
    echo "Note: This will be slower but still functional."
    docker-compose -f docker-compose.cpu.yml up -d
fi

# Wait for the application to become available
echo ""
echo "⏳ Waiting for the application to start (this may take a minute)..."
echo "   • First-time startup will be slower as models are downloaded"
echo "   • You can check progress with: docker-compose logs -f webapp"
echo ""

for i in {1..45}; do
    if curl -s http://localhost:8501 > /dev/null; then
        echo ""
        echo "✅ Application is ready!"
        echo ""
        echo "   📄 Access the RAG Pipeline at: http://localhost:8501"
        echo ""
        echo "   1. First navigate to the 'Upload Documents' tab"
        echo "   2. Upload your documents (PDF, DOCX, CSV, etc.)"
        echo "   3. Click 'Process Documents' to extract and index the content"
        echo "   4. Go to the 'Ask Questions' tab to query your documents"
        echo ""
        exit 0
    fi
    sleep 3
done

echo "⚠️ The application seems to be taking longer than expected to start."
echo "It may still be initializing in the background."
echo ""
echo "🔗 Once ready, you can access it at: http://localhost:8501"
echo ""
echo "📋 To view startup progress:"
echo "   docker-compose logs -f webapp"
echo "" 