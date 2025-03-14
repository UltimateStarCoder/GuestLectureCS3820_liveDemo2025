# Simple RAG Pipeline with Ollama

A simple Retrieval-Augmented Generation (RAG) pipeline that demonstrates the key components of RAG in under 100 lines of code, powered by Ollama. This project is designed for educational purposes to help students understand how RAG works.

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models by providing them with relevant external knowledge. RAG consists of three main components:

1. **Retrieval**: Finding relevant information from a knowledge base
2. **Augmentation**: Adding this information to the context of the query
3. **Generation**: Using an LLM to generate a response based on the augmented context

This technique significantly improves accuracy, especially for domain-specific or factual questions, by grounding the LLM's responses in relevant documents.

## Features

- **Complete RAG Pipeline**: Implements all components in a minimal, easy-to-understand way
- **Universal Document Processing**: Process virtually any document format:
  - Text Files (txt, md)
  - PDF Documents (pdf)
  - Microsoft Office (doc, docx, ppt, pptx, xls, xlsx)
  - OpenDocument Formats (odt, odp, ods)
  - Data Files (csv, json, xml)
  - Web Files (html, htm)
- **Vector Database**: Store and retrieve document chunks using semantic similarity
- **Edge-Ready LLM**: Uses TinyLlama (1.1B parameters) for fast responses on limited hardware
- **Docker Integration**: Run everything in containers for easy setup
- **Educational Value**: Clean, well-commented code for learning purposes

## Prerequisites

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)
- **Hardware Requirements**:
  - Minimum: 2GB RAM, dual-core CPU
  - Recommended: 4GB RAM, quad-core CPU
  - GPU support is optional but beneficial

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd simple-rag-pipeline

# Start the application (with GPU support if available)
docker-compose up

# For CPU-only environments
docker-compose -f docker-compose.cpu.yml up
```

### Usage

1. **Access the application**:
   - Open your browser and go to: http://localhost:8501

2. **Process documents**:
   - Navigate to the "Upload Documents" tab
   - Upload documents in any supported format
   - Click "Process Documents" to extract, chunk, and index the content

3. **Ask questions**:
   - Switch to the "Ask Questions" tab
   - Enter a question related to your documents
   - Click "Get Answer" to retrieve relevant information and generate a response

4. **Review answers**: 
   - The answer is generated based on the most relevant portions of your documents
   - The system uses semantic search to find the most relevant content

## How It Works

### System Architecture

The application is built with these key components:

- **Frontend**: Streamlit provides a simple, interactive web UI
- **Document Processing**: Multiple libraries for handling different document formats:
  - PyPDF2 for PDF files
  - docx2txt for Word documents
  - pandas for spreadsheets and CSV files
  - BeautifulSoup for HTML/XML
  - LibreOffice for other office documents
- **Text Splitting**: LangChain's RecursiveCharacterTextSplitter divides documents into manageable chunks
- **Embedding Generation**: Sentence Transformers convert text into vector embeddings
- **Vector Database**: ChromaDB stores and retrieves document vectors
- **LLM Integration**: Ollama serves TinyLlama for edge-optimized answer generation
- **Orchestration**: LangChain connects all components together

### RAG Pipeline Steps

1. **Document Ingestion**:
   - Documents are uploaded through the web interface
   - Text is extracted using format-specific processors
   - The text is split into smaller, semantically meaningful chunks
   - Each chunk is converted to a vector embedding and stored in ChromaDB

2. **Query Processing**:
   - The user's question is converted to a vector embedding
   - The system performs similarity search to find the most relevant document chunks
   - Retrieved chunks provide context for the LLM

3. **Answer Generation**:
   - The question and retrieved context are sent to the TinyLlama model via Ollama
   - The model generates an answer based solely on the provided context
   - The answer is displayed to the user

## Technical Details

### File Structure

- `app.py`: Main application code implementing the RAG pipeline
- `Dockerfile`: Container configuration for the application
- `docker-compose.yml`: Multi-container setup with GPU support
- `docker-compose.cpu.yml`: Alternative setup for CPU-only environments
- `init-ollama.sh`: Initialization script for model downloading
- `requirements.txt`: Python dependencies

### Models Used

- **Embedding Model**: all-MiniLM-L6-v2 (small, efficient embedding model)
- **Language Model**: TinyLlama (1.1B parameter model optimized for edge devices)

## Customization

You can customize this RAG pipeline in several ways:

- **Change the LLM model**: Edit `app.py` and `init-ollama.sh` to use a different Ollama model
- **Adjust chunk size**: Modify the `chunk_size` parameter in `app.py` (larger chunks provide more context)
- **Add document formats**: Extend the `process_document` function to support additional file types
- **Modify the prompt**: Change the `template` in the `PromptTemplate` to guide the LLM's responses

## Troubleshooting

- **Docker Issues**: Ensure Docker and Docker Compose are properly installed and running
- **GPU Problems**: If you encounter GPU-related errors, try using the CPU version:
  ```bash
  docker-compose -f docker-compose.cpu.yml up
  ```
- **Document Processing Issues**: Check that required system libraries are installed in the Dockerfile
- **Port Conflicts**: If port 8501 is already in use, modify the port mapping in the docker-compose file
- **Memory Issues**: Even with the lightweight TinyLlama model, you still need at least 2GB of RAM

## Educational Resources

To learn more about RAG and the technologies used in this project:

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [Understanding Vector Databases](https://www.pinecone.io/learn/vector-database/)
- [Ollama Documentation](https://ollama.ai/documentation)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

This project is provided as an educational resource under the MIT License. 