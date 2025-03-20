#!/bin/bash
# Reset RAG Pipeline with FAISS
# This script resets the system by:
# 1. Stopping any running containers
# 2. Removing Docker volumes to clear FAISS indices and documents
# 3. Rebuilding and starting the containers

# ANSI color codes for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print a colorful header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}      RAG Pipeline with FAISS Reset Tool    ${NC}"
echo -e "${BLUE}============================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Error: Docker is not running.${NC}"
  echo -e "${YELLOW}Please start Docker Desktop or Docker Engine first.${NC}"
  exit 1
fi

# Check if Ollama is installed and running
echo -e "${YELLOW}Checking if Ollama is installed on your host machine...${NC}"
if ! curl -s --max-time 3 --head http://localhost:11434/api/tags > /dev/null; then
  echo -e "${RED}Warning: Cannot connect to Ollama on your host machine.${NC}"
  echo -e "${YELLOW}This RAG application requires Ollama to be installed and running on your host.${NC}"
  echo -e "${YELLOW}Please install Ollama from https://ollama.ai/download and start it.${NC}"
  
  # Ask if user wants to continue anyway
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Exiting. Please install Ollama and try again.${NC}"
    exit 1
  fi
else
  echo -e "${GREEN}✓ Ollama is running on your host machine.${NC}"
  
  # Check if the required model is available
  echo -e "${YELLOW}Checking if llama3:8b model is available...${NC}"
  if ! curl -s --max-time 3 -X POST "http://localhost:11434/api/show" -d '{"name":"llama3:8b"}' | grep -q "model"; then
    echo -e "${RED}Warning: The llama3:8b model is not available in your Ollama installation.${NC}"
    echo -e "${YELLOW}You should download it by running this command: ollama pull llama3:8b${NC}"
    
    # Ask if user wants to download it now
    read -p "Do you want to download the model now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      echo -e "${YELLOW}Downloading llama3:8b model (this may take several minutes)...${NC}"
      ollama pull llama3:8b
      if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download model. Please run 'ollama pull llama3:8b' manually.${NC}"
      else
        echo -e "${GREEN}✓ Model downloaded successfully.${NC}"
      fi
    fi
  else
    echo -e "${GREEN}✓ llama3:8b model is available.${NC}"
  fi
fi

# Clean up any leftover containers
echo -e "\n${YELLOW}Stopping any running containers...${NC}"
docker-compose down --remove-orphans 2>/dev/null
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Containers stopped successfully.${NC}"
else
  echo -e "${YELLOW}No containers were running or docker-compose failed.${NC}"
fi

# Remove Docker volumes to completely reset data
echo -e "\n${YELLOW}Removing Docker volumes to clear all data...${NC}"
docker volume rm $(docker volume ls -q | grep "faiss_data\|doc_storage") 2>/dev/null
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Docker volumes removed successfully. All FAISS indices and documents have been cleared.${NC}"
else
  echo -e "${YELLOW}No matching volumes found or they could not be removed.${NC}"
fi

# Build and start containers
echo -e "\n${YELLOW}Building and starting containers with fresh volumes...${NC}"
docker-compose up --build -d --remove-orphans

if [ $? -eq 0 ]; then
  echo -e "${GREEN}Containers started successfully.${NC}"
  
  # Check if services are ready
  echo -e "${YELLOW}Waiting for services to initialize...${NC}"
  
  # Check Streamlit readiness
  MAX_RETRIES=30
  RETRY_COUNT=0
  echo -e "${YELLOW}Checking Streamlit service...${NC}"
  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s --head http://localhost:8501 > /dev/null; then
      echo -e "${GREEN}Streamlit service is ready!${NC}"
      STREAMLIT_READY=true
      break
    fi
    echo -n "."
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT+1))
  done
  
  if [ "$STREAMLIT_READY" != "true" ]; then
    echo -e "\n${YELLOW}Warning: Streamlit service did not respond within 30 seconds.${NC}"
    echo -e "${YELLOW}  - Check status: ${NC}docker-compose ps"
    echo -e "${YELLOW}  - View logs: ${NC}docker-compose logs -f"
    echo -e "${YELLOW}  - Service may still become available shortly${NC}"
  fi
  
  # Display app URL
  echo -e "\n${BLUE}============================================${NC}"
  echo -e "${GREEN}RAG Pipeline with FAISS is now running!${NC}"
  echo -e "${GREEN}Access the application at: ${BLUE}http://localhost:8501${NC}"
  echo -e "${BLUE}============================================${NC}"
  
  # Display logs option
  echo -e "\n${YELLOW}To view logs, run: ${NC}docker-compose logs -f"
  echo -e "${YELLOW}To stop the application, run: ${NC}docker-compose down\n"
else
  echo -e "${RED}Failed to start containers. Check the error messages above.${NC}"
  exit 1
fi 