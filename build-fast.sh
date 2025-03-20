#!/bin/bash
# Optimized script for fast Docker builds
# This script:
# 1. Enables BuildKit for faster, parallel, and cached builds
# 2. Ensures proper environment variables for optimal Docker builds
# 3. Builds the Docker image with optimized settings
# 4. Provides clear status feedback on build progress

# Print colored messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
RED='\033[0;31m'

# Set strict error handling
set -e

echo -e "${BLUE}=== Building optimized Docker image ===${NC}"

# Enable Docker BuildKit for faster parallel builds
# BuildKit provides:
# - Concurrent dependency resolution
# - Enhanced caching capabilities
# - More efficient build process
# - Better layer management
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
export BUILDKIT_PROGRESS=plain

# Function for status reporting
report_status() {
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ $1${NC}"
  else
    echo -e "${RED}❌ $1${NC}"
    exit 1
  fi
}

# Optional: Prune unused build cache - uncomment if needed
# echo -e "${YELLOW}Cleaning up old build cache...${NC}"
# docker builder prune -f --filter until=24h

# Build with optimized settings
echo -e "${YELLOW}Building with optimized settings...${NC}"
docker-compose build --parallel --pull --build-arg BUILDKIT_INLINE_CACHE=1
report_status "Build completed successfully!"

# Provide next steps for the user
echo -e "${BLUE}To run: docker-compose up -d${NC}"
