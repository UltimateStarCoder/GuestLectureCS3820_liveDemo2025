# Simple RAG Pipeline

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

### Docker Setup
1. **Install Docker Desktop**:
   - Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - This includes both Docker Engine and Docker Compose
   - Available for Windows, macOS, and Linux

2. **Start Docker Desktop**:
   - Launch Docker Desktop application
   - Wait for it to fully start (you'll see the Docker icon in your system tray)
   - The Docker engine must be running in the background before using this RAG pipeline

### Hardware Requirements
- **Minimum**: 2GB RAM, dual-core CPU
- **Recommended**: 4GB RAM, quad-core CPU

## Quick Start

### Installation and Startup

```bash
# Clone the repository
git clone <repository-url>
cd GuestLectureCS3820_liveDemo2025

# Make the reset script executable (REQUIRED for macOS/Linux)
chmod +x reset_rag.sh

# Start the application (will also clear any existing data)
./reset_rag.sh
```

> **Note for macOS/Linux Users**: The `chmod +x reset_rag.sh` command is required to make the script executable. Without this step, you'll get a "Permission denied" error when trying to run the script.
>
> **Note for Windows Users**: If using WSL (Windows Subsystem for Linux), follow the Unix instructions above. If using PowerShell or Command Prompt, you may need to run `sh reset_rag.sh` instead.

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

4. **Clear data or restart**:
   - Option 1: Click the "Clean Vector Store" button in the UI to clear the database and all processed documents without restarting the application
   - Option 2: Run `./reset_rag.sh` to clear all data and restart the application

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
- `docker-compose.yml`: Multi-container setup for Docker
- `init-ollama.sh`: Initialization script for model downloading
- `reset_rag.sh`: Script to reset data and restart the application
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

- **Docker Issues**: 
  - Ensure Docker Desktop is installed and running
  - Check that the Docker engine is active (Docker Desktop icon should be running)
  - If Docker Desktop isn't running, the RAG pipeline won't work
- **Document Processing Issues**: Check that required system libraries are installed in the Dockerfile
- **Port Conflicts**: If port 8501 is already in use, modify the port mapping in the docker-compose file
- **Memory Issues**: Even with the lightweight TinyLlama model, you still need at least 2GB of RAM
- **Log Viewing**: Check logs with `docker-compose logs -f webapp` for detailed debugging

## Educational Resources

To learn more about RAG and the technologies used in this project:

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [Understanding Vector Databases](https://www.pinecone.io/learn/vector-database/)
- [Ollama Documentation](https://ollama.ai/documentation)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

This project is provided as Open Source software under the MIT License.

### MIT License

Copyright (c) 2025 Mac-Rufus O. Umeokolo

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

**Note:** By using this software, you are agreeing to the terms of the MIT License as specified above. You must read and accept this license before using the application. 