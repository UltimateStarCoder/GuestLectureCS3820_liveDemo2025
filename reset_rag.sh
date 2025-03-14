#!/bin/bash

# ANSI color codes for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print a step with color
print_step() {
  echo -e "${BLUE}==> ${1}${NC}"
}

# Function to print success message
print_success() {
  echo -e "${GREEN}✓ ${1}${NC}"
}

# Function to print warning
print_warning() {
  echo -e "${YELLOW}! ${1}${NC}"
}

# Function to print error
print_error() {
  echo -e "${RED}✗ ${1}${NC}"
}

print_header() {
  echo -e "${CYAN}"
  echo "╔══════════════════════════════════════════╗"
  echo "║     RAG System Reset and Start Script     ║"
  echo "╚══════════════════════════════════════════╝"
  echo -e "${NC}"
}

# Display license information and request acceptance
display_license() {
  echo -e "${BOLD}LICENSE INFORMATION:${NC}"
  echo "This is Open Source software under the MIT License."
  echo "Copyright (c) 2025 Mac-Rufus O. Umeokolo"
  echo ""
  echo "By proceeding, you are agreeing to the terms of the MIT License."
  echo "For the full license text, see the README.md or LICENSE file."
  echo ""
  echo -e "${YELLOW}You must accept the license to use this application.${NC}"
  echo ""
  read -p "Do you accept the terms of the MIT License? (y/n): " license_response
  
  # Convert to lowercase
  license_response=$(echo "$license_response" | tr '[:upper:]' '[:lower:]')
  
  if [ "$license_response" != "y" ] && [ "$license_response" != "yes" ]; then
    print_error "License not accepted. Exiting."
    exit 1
  fi
  
  print_success "License accepted. Proceeding with application setup."
  echo ""
}

print_header
display_license

# Set the standard docker-compose file
print_step "Configuring RAG system..."
print_success "Using optimized configuration for Docker"
COMPOSE_FILE="docker-compose.yml"

# Stop all containers
print_step "Stopping all containers..."
docker-compose down --remove-orphans
if [ $? -eq 0 ]; then
  print_success "Containers stopped successfully"
else
  print_error "Failed to stop containers"
  exit 1
fi

# Create data directories if they don't exist
print_step "Ensuring data directories exist..."
mkdir -p ./data/chroma
mkdir -p ./documents
print_success "Data directories ready"

# Clear vector store
print_step "Clearing vector store..."
if [ -d "./data/chroma" ]; then
  # Check if directory is empty
  if [ "$(ls -A ./data/chroma 2>/dev/null)" ]; then
    # Directory exists and has content
    find ./data/chroma -type f -delete
    find ./data/chroma -type d -not -path "./data/chroma" -delete
    print_success "Vector store data cleared"
  else
    print_warning "Vector store directory is already empty"
  fi
else
  print_warning "Vector store directory doesn't exist, creating it"
  mkdir -p ./data/chroma
fi

# Clear documents
print_step "Clearing documents..."
if [ -d "./documents" ]; then
  # Check if directory is empty
  if [ "$(ls -A ./documents 2>/dev/null)" ]; then
    # Directory exists and has content
    find ./documents -type f -delete
    print_success "Documents cleared"
  else
    print_warning "Documents directory is already empty"
  fi
else
  print_warning "Documents directory doesn't exist, creating it"
  mkdir -p ./documents
fi

# Start containers again
print_step "Starting RAG system..."
docker-compose up -d
if [ $? -eq 0 ]; then
  print_success "Containers started successfully"
else
  print_error "Failed to start containers"
  exit 1
fi

# Wait for the application to become available
echo ""
print_step "Waiting for RAG application to be ready..."
echo "First-time startup may take up to a minute..."
echo ""

for i in {1..45}; do
  if curl -s http://localhost:8501 > /dev/null; then
    echo ""
    print_success "RAG system is ready!"
    echo ""
    echo -e "${GREEN}   📄 Access the RAG Pipeline at: http://localhost:8501${NC}"
    echo ""
    echo "   1. Upload your documents in the 'Upload Documents' tab"
    echo "   2. Process them with the 'Process Documents' button"
    echo "   3. Ask questions in the 'Ask Questions' tab"
    echo "   4. To clear existing data, use the 'Clean Vector Store' button"
    echo ""
    echo -e "Using: ${CYAN}TinyLlama model${NC} optimized for lightweight usage"
    echo ""
    echo -e "${YELLOW}This software is licensed under the MIT License"
    echo -e "Copyright (c) 2025 Mac-Rufus O. Umeokolo${NC}"
    exit 0
  fi
  echo -n "."
  sleep 2
done

echo ""
print_warning "Application is taking longer than expected to start."
print_warning "Check logs with: docker-compose logs -f webapp"
print_warning "Once ready, access at: http://localhost:8501" 