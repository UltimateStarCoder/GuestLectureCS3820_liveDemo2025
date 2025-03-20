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

# Clean up any leftover containers
echo -e "\n${YELLOW}Stopping any running containers...${NC}"
docker-compose down 2>/dev/null
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
docker-compose up --build -d

if [ $? -eq 0 ]; then
  echo -e "${GREEN}Containers started successfully.${NC}"
  
  # Wait a moment for the services to initialize
  echo -e "${YELLOW}Waiting for services to initialize...${NC}"
  sleep 5
  
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