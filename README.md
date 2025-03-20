# Minimal RAG Pipeline with FAISS Vector Store

A containerized Retrieval-Augmented Generation (RAG) pipeline that uses FAISS for efficient vector search and Ollama for local LLM inference. This application allows you to upload documents in various formats, process them into embeddings, and query them using natural language.

## 🚀 Features

- **Document Processing**: Support for multiple document formats (PDF, DOCX, TXT, CSV, JSON, etc.)
- **FAISS Vector Search**: High-performance similarity search using Facebook AI Similarity Search
- **Ollama LLM Integration**: Local inference with Ollama LLM models
- **Containerized Deployment**: Fully Dockerized for easy deployment
- **Streamlit UI**: Simple, intuitive interface for document upload and querying
- **Persistence**: FAISS indices are stored on disk and persist between application restarts

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
- **Ollama**: For local LLM inference
- **Streamlit**: For the web interface
- **Docker**: For containerization
- **HuggingFace Embeddings**: For generating vector embeddings

## 🔧 Installation & Deployment

### Prerequisites

- Docker and Docker Compose

### Deployment Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run with Docker Compose:
   ```bash
   docker-compose up --build -d
   ```

3. Access the application:
   Open your browser and navigate to http://localhost:8501

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
   This script will:
   - Stop all running containers
   - Remove Docker volumes to clear FAISS indices and documents
   - Rebuild and restart the containers with fresh volumes

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

1. **Clear the Vector Store**: Use the "Clear Vector Store" button in the UI to reset the FAISS index
2. **Reset All Data**: Run `./reset_rag.sh` to completely reset all Docker volumes
3. **View Logs**: Check container logs with `docker-compose logs -f`
4. **Restart Containers**: Restart the application with `docker-compose restart`

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
├── data/                  # Directory for data storage
│   └── faiss/             # FAISS index storage
├── documents/             # Uploaded document storage
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile             # Docker build configuration
├── init-ollama.sh         # Ollama initialization script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 📊 Performance Considerations

- **Document Size**: Large documents are split into chunks for better processing
- **Caching**: Multiple layers of caching improve performance
- **FAISS Configuration**: Default configuration works well for most use cases

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