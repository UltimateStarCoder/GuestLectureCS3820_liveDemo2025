# Minimal RAG Pipeline with FAISS Vector Store

A containerized Retrieval-Augmented Generation (RAG) pipeline that uses FAISS for efficient vector search and Ollama for local LLM inference with the Llama 3 8B model. This application allows you to upload documents in various formats, process them into embeddings, and query them using natural language.

## 🚀 Features

- **Document Processing**: Support for multiple document formats (PDF, DOCX, TXT, CSV, JSON, LaTeX, etc.)
- **FAISS Vector Search**: High-performance similarity search using Facebook AI Similarity Search
- **Llama 3 8B Integration**: Local inference with Ollama running the Llama 3 8B model
- **Hybrid Deployment**: Application is containerized while leveraging host's Ollama installation for maximum LLM performance
- **Streamlit UI**: Simple, intuitive interface for document upload and querying
- **Persistence**: FAISS indices are stored on disk and persist between application restarts
- **LaTeX Support**: Process LaTeX documents including mathematical expressions

## 📋 Architecture

The RAG pipeline consists of the following components:

1. **Document Processing**: Extract text from various document formats
2. **Text Chunking**: Split documents into manageable pieces using RecursiveCharacterTextSplitter
3. **Vector Embedding**: Generate embeddings using HuggingFace's sentence-transformers
4. **FAISS Indexing**: Store and retrieve vectors efficiently using FAISS
5. **Retrieval**: Find the most similar document chunks for a given query
6. **Generation**: Generate answers using Ollama LLM

## 💻 Technical Stack

- **FAISS**: For efficient vector similarity search
- **LangChain**: For orchestrating the RAG pipeline
- **Ollama with Llama 3 8B**: For local LLM inference
- **Streamlit**: For the web interface
- **Docker**: For containerization
- **HuggingFace Embeddings**: For generating vector embeddings

## 🔧 Installation & Deployment

### Ollama Requirements

This application requires Ollama to be installed and running on your host machine:

1. **Install Ollama**:
   - Download from [ollama.ai/download](https://ollama.ai/download)
   - Install the application for your operating system
   - Start the Ollama application (look for the icon in your system tray/menu bar)

2. **Download the Required Model**:
   - Open a terminal/command prompt
   - Run: `ollama pull llama3:8b`
   - Wait for the model to download (this may take several minutes)

3. **Verify Installation**:
   - Run: `ollama list` to see installed models
   - Ensure `llama3:8b` appears in the list

### Prerequisites

- Docker and Docker Compose
- Ollama installed on your host machine
- Llama 3 8B model pulled in Ollama (`ollama pull llama3:8b`)

### Optimized Docker Deployment

The project uses a performance-optimized Docker setup for faster builds and efficient resource usage:

1. Install Ollama and pull the model as described in the Ollama Requirements section

2. Clone the repository and prepare the build scripts:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   chmod +x build-fast.sh reset_rag.sh
   ```

3. Build using the optimized build script:
   ```bash
   ./build-fast.sh
   ```

4. Start the application:
   ```bash
   docker-compose up -d
   ```

5. Access the application:
   Open your browser and navigate to http://localhost:8501

This deployment offers several advantages:
- Multi-stage Docker builds for faster build times
- Better layer caching for improved performance
- Optimized resource usage with container limits
- Direct access to local GPU resources if available
- Simpler model management through Ollama Desktop

## 🚀 How to Use

### Working with the Application

1. **Start the Application**:
   ```bash
   docker-compose up -d
   ```

2. **Upload Documents**:
   - Navigate to the "Upload Documents" tab
   - Upload one or more supported documents
   - Click "Process Documents" to create vector embeddings
   - Your documents are now indexed and stored in FAISS

3. **Query Your Documents**:
   - Navigate to the "Ask Questions" tab
   - Type your question in natural language
   - Click "Get Answer" to get a response based on your documents

4. **Reset All Data**:
   ```bash
   ./reset_rag.sh
   ```
   This optimized reset script will:
   - Stop all running containers
   - Clear Docker volumes to reset FAISS indices and documents
   - Rebuild and restart the containers with optimized settings
   - Provide clear status feedback throughout the process

5. **View Performance Metrics**:
   - Navigate to the "Performance" tab
   - See statistics on document processing, embedding generation, and query times

### Data Management

All data is stored in Docker volumes for complete isolation:

- `faiss_data`: Stores FAISS indices in `/app/data`
- `doc_storage`: Stores uploaded documents in `/app/documents`
- `ollama_data`: Stores downloaded LLM models

These volumes are managed entirely by Docker, ensuring:
- Proper data isolation
- Easy backup and restore
- Simple reset process

### Troubleshooting

If you encounter issues:

1. **Check Ollama Status**: 
   - Go to the "System Status" tab in the application
   - It will show if Ollama is running and if the model is available
   - Follow the suggested steps if there are any issues

2. **Clear the Vector Store & Uploads**: 
   - Use the "Clear Vector Store & Uploads" button in the UI to reset the FAISS index and remove all uploaded documents

3. **Verify Ollama on Host Machine**:
   - Open a terminal and run `ollama list` to see installed models
   - Run `curl http://localhost:11434/api/tags` to verify the Ollama API is accessible
   - If using a different port, update the OLLAMA_HOST environment variable in docker-compose.yml

4. **Restart Services**:
   - Restart Ollama application on your host machine
   - Restart the containerized app with `docker-compose restart`

5. **Check Logs**:
   - View container logs with `docker-compose logs -f`
   - Look for any error messages related to Ollama connections

6. **Network Issues**:
   - If using Docker Desktop, ensure host.docker.internal resolves correctly
   - If running on Linux, you might need to use the host machine's actual IP instead of host.docker.internal

7. **Permission Issues**: 
   - If you can't execute scripts, run `chmod +x script_name.sh` to make them executable

## 🔍 Why FAISS?

FAISS (Facebook AI Similarity Search) offers several advantages for RAG applications:

- **Performance**: Optimized C++ implementation for fast similarity search
- **Memory Efficiency**: Specialized data structures for efficient vector storage
- **Scalability**: Can handle millions of vectors with minimal performance degradation
- **Configurability**: Multiple index types (HNSW, IVF, etc.) for different use cases
- **Active Development**: Well-maintained by Meta AI Research

## 📁 Directory Structure

```
.
├── app.py                 # Main application code
├── build-fast.sh          # Optimized Docker build script
├── data/                  # Directory for data storage
│   └── faiss/             # FAISS index storage
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile             # Multi-stage Docker build configuration
├── documents/             # Uploaded document storage
├── requirements.txt       # Python dependencies
├── reset_rag.sh           # System reset script
├── startup.sh             # Container startup script
├── QUICKSTART.txt         # Quick start guide
└── README.md              # This file
```

## 🔧 Docker Optimization Details

The project uses several Docker optimization techniques:

1. **Multi-Stage Builds**: The Dockerfile uses a three-stage build process:
   - Base stage for system dependencies
   - Dependencies stage for Python packages
   - Final stage with just the application code

2. **Layer Caching**: Carefully structured to maximize Docker layer cache usage:
   - Dependencies installed separately from application code
   - Requirements copied before code to avoid unnecessary rebuilds

3. **BuildKit Integration**: Uses Docker BuildKit for faster, more efficient builds:
   - Parallel dependency resolution
   - Better caching mechanisms
   - More efficient layer storage

4. **Resource Management**: Container resource limits prevent resource contention:
   - Memory limits prevent excessive RAM usage
   - CPU limits ensure stable performance
   - Health checks monitor application status

5. **Volume Optimization**: Uses read-only volume mounts for security:
   - Application code mounted read-only
   - Data volumes properly isolated

To take advantage of these optimizations, always use the provided scripts:
- `./build-fast.sh` for building the Docker image
- `./reset_rag.sh` for resetting the application state

## 📊 Performance Considerations

- **Document Size**: Large documents are split into chunks for better processing
- **Caching**: Multiple layers of caching improve performance
- **FAISS Configuration**: Default configuration works well for most use cases
- **LLM Model**: Using Llama 3 8B for a good balance between performance and efficiency
- **Hybrid Deployment**: The containerized app connecting to host Ollama provides better LLM performance than a fully containerized solution

## 🔒 Security Notes

- This application is designed for local deployment or trusted environments
- No authentication is implemented in the basic version
- Documents and their vectors are stored locally

## 📄 License

MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgements

- FAISS by Meta Research
- LangChain for the RAG framework
- Ollama for local LLM inference
- Streamlit for the web interface 