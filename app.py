"""
Minimal RAG Pipeline with FAISS Vector Store

This application implements a Retrieval-Augmented Generation (RAG) pipeline
using FAISS (Facebook AI Similarity Search) for efficient vector storage and retrieval.
The application allows users to upload documents, process them into chunks,
create vector embeddings, and query the indexed documents using natural language.

Key components:
- Document processing: Extract text from various document formats
- Text chunking: Split text into manageable pieces
- Embedding generation: Convert text chunks to vector embeddings using HuggingFace models
- Vector storage: Store and retrieve vectors efficiently using FAISS
- Answer generation: Generate answers to user questions using Ollama LLM

The application is containerized with Docker and designed for deployment
in resource-constrained environments.
"""

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import tempfile
import docx2txt
import csv
import json
import pandas as pd
import re
from bs4 import BeautifulSoup
from pathlib import Path
import io
import shutil
import subprocess
import hashlib
import time
from unstructured.partition.auto import partition
from langchain.schema import Document
import requests
# Import faiss for direct binary serialization
import faiss
import numpy as np
import pickle
import sys
# Import LaTeX processing libraries
from pylatexenc.latex2text import LatexNodes2Text

# Get Ollama host from environment variable, default to localhost
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:11434"
REQUIRED_MODEL = "llama3:8b"

# Function to check if the LLM is loaded and ready
def check_ollama_status():
    """
    Comprehensive check of Ollama installation and model status.
    
    Returns:
        dict: Status information with keys:
            - ollama_running: bool - If Ollama server is accessible
            - model_available: bool - If the required model is available
            - model_name: str - Name of the required model
            - error_message: str - Error details if any
            - version: str - Ollama version if available
            - host: str - Host being used
    """
    status = {
        "ollama_running": False,
        "model_available": False,
        "model_name": REQUIRED_MODEL,
        "error_message": None,
        "version": "unknown",
        "host": OLLAMA_HOST
    }
    
    try:
        # Check if Ollama server is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        
        if response.status_code == 200:
            status["ollama_running"] = True
            
            # Try to get Ollama version
            try:
                version_response = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=2)
                if version_response.status_code == 200:
                    status["version"] = version_response.json().get("version", "unknown")
            except:
                pass  # Ignore version check failures
            
            # Check if the required model is available
            models = response.json().get("models", [])
            for model in models:
                if REQUIRED_MODEL.lower() in model.get("name", "").lower():
                    status["model_available"] = True
                    break
            
            # If model not found in list, try direct model info check
            if not status["model_available"]:
                try:
                    model_response = requests.post(
                        f"{OLLAMA_BASE_URL}/api/show",
                        json={"name": REQUIRED_MODEL},
                        timeout=5
                    )
                    if model_response.status_code == 200:
                        status["model_available"] = True
                except Exception as e:
                    status["error_message"] = f"Model check error: {str(e)}"
        else:
            status["error_message"] = f"Ollama server returned status code: {response.status_code}"
            
    except requests.ConnectionError:
        status["error_message"] = f"Cannot connect to Ollama at {OLLAMA_BASE_URL}"
    except requests.Timeout:
        status["error_message"] = f"Connection to Ollama at {OLLAMA_BASE_URL} timed out"
    except Exception as e:
        status["error_message"] = f"Unexpected error: {str(e)}"
    
    return status

def check_llm_status():
    """
    Simple check if the LLM is ready to use.
    
    Returns:
        bool: True if Ollama is running and the model is available
    """
    status = check_ollama_status()
    return status["ollama_running"] and status["model_available"]

def display_ollama_status():
    """
    Display the Ollama status in the Streamlit UI with helpful guidance.
    """
    status = check_ollama_status()
    
    st.subheader("Ollama Connection Status")
    
    # Show refresh button
    if st.button("🔄 Refresh Status"):
        st.experimental_rerun()
    
    # Show connection info
    st.info(f"Connecting to Ollama at: {OLLAMA_BASE_URL}")
    
    if status["ollama_running"] and status["model_available"]:
        st.success(f"✅ Ollama (v{status['version']}) is running with {status['model_name']} model ready")
        st.balloons()  # A little celebration
        return True
    
    # Ollama not running
    if not status["ollama_running"]:
        st.error(f"❌ Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        st.info("Please check that:")
        st.markdown("""
        1. **Ollama is installed on your computer**
           - Download from: [https://ollama.ai/download](https://ollama.ai/download)
        2. **Ollama application is running**
           - Look for the Ollama icon in your system tray/menu bar
        3. **Ollama is listening on the default port (11434)**
           - You can verify by opening a browser and going to: http://localhost:11434
        """)
        
        with st.expander("Why do I need Ollama?"):
            st.markdown("""
            This RAG application uses Ollama to run the Llama 3 language model locally on your machine.
            This provides better privacy (your data stays local) and eliminates the need for API keys.
            """)
    
    # Ollama running but model not available
    elif not status["model_available"]:
        st.warning(f"⚠️ Ollama is running, but the **{status['model_name']}** model is not available")
        
        st.info("You need to pull (download) the model from Ollama's model library.")
        
        st.markdown("### Option 1: Terminal Command")
        st.markdown("Run this command in your terminal to download the model:")
        st.code(f"ollama pull {status['model_name']}")
        
        st.markdown("### Option 2: Pull from Ollama App")
        st.markdown("Open the Ollama application and search for the model in the library.")
        
        if st.button("Try Automatic Pull (Experimental)"):
            with st.spinner(f"Pulling {status['model_name']} model (this may take several minutes)..."):
                try:
                    # Try to pull the model
                    pull_response = requests.post(
                        f"{OLLAMA_BASE_URL}/api/pull",
                        json={"name": REQUIRED_MODEL},
                        timeout=10  # Just start the pull, don't wait for completion
                    )
                    
                    if pull_response.status_code == 200:
                        st.success("✅ Model pull started successfully!")
                        st.info("This process will continue in the background and may take several minutes.")
                        st.info("Click 'Refresh Status' button to check if the model is ready.")
                    else:
                        st.error(f"Failed to start model pull. Status code: {pull_response.status_code}")
                except Exception as e:
                    st.error(f"Error during model pull: {str(e)}")
        
        with st.expander("About this model"):
            st.markdown(f"""
            **{status['model_name']}** is the 8 billion parameter version of Llama 3, Meta's latest 
            large language model. It's a smaller, faster version of Llama 3 that still provides
            excellent performance for most tasks.
            
            * Size: ~5GB download
            * Memory usage: ~8GB RAM when running
            * Capabilities: Text generation, Q&A, summarization, and more
            """)
    
    # Show environment details for debugging
    with st.expander("Environment Details"):
        st.code(f"""
OLLAMA_HOST: {OLLAMA_HOST}
OLLAMA_BASE_URL: {OLLAMA_BASE_URL}
Python Version: {sys.version}
OS: {os.name}
Docker: {"Yes (containerized)" if os.path.exists("/.dockerenv") else "No (native)"}
        """)
    
    # Show any error message
    if status["error_message"]:
        with st.expander("Error details"):
            st.text(status["error_message"])
    
    return False

###########################################
# INITIALIZATION AND CONFIGURATION
###########################################

# Setup directories for storing data
DOCS_DIR = "./documents"
FAISS_INDEX_DIR = "./data/faiss"  # Directory to store FAISS indices
os.makedirs(DOCS_DIR, exist_ok=True)    # Create directory for document storage
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)  # Create directory for FAISS indices

# Initialize Streamlit session state to persist data between reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
# Track processed files to avoid reprocessing
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

# Track performance metrics
if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = {
        "document_processing_time": 0,
        "embedding_generation_time": 0,
        "query_time": 0
    }

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings_model():
    """
    Load and cache the sentence embedding model.
    This prevents reloading the model on every rerun, which is expensive.
    
    Returns:
        HuggingFaceEmbeddings: The embedding model instance
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# We use a small but effective model from Hugging Face
embeddings = get_embeddings_model()

# Define supported file extensions
SUPPORTED_EXTENSIONS = {
    "pdf": "Adobe PDF",
    "txt": "Text File",
    "docx": "Word Document",
    "doc": "Word Document (older format)",
    "csv": "CSV Spreadsheet",
    "xlsx": "Excel Spreadsheet",
    "xls": "Excel Spreadsheet (older format)",
    "pptx": "PowerPoint Presentation",
    "ppt": "PowerPoint Presentation (older format)",
    "json": "JSON File",
    "md": "Markdown",
    "html": "HTML File",
    "htm": "HTML File",
    "xml": "XML File",
    "rtf": "Rich Text Format",
    "odt": "OpenDocument Text",
    "tex": "LaTeX Document",
    "ltx": "LaTeX Document",
    "latex": "LaTeX Document"
}

# Extensions that may need LibreOffice for conversion
LIBREOFFICE_EXTENSIONS = [
    "doc", "xls", "ppt", "odt", "ods", "odp", "rtf"
]

# File extensions that may work with Unstructured but with potential limitations
UNSTRUCTURED_COMPATIBLE = [
    "pdf", "docx", "pptx", "html", "htm", "md", "xlsx"
]

###########################################
# DOCUMENT PROCESSING FUNCTIONS
###########################################

@st.cache_data(show_spinner=False)
def extract_text_with_libreoffice(temp_file_path):
    """
    Use LibreOffice to convert documents to text format.
    Useful for handling various office document formats.
    
    Args:
        temp_file_path: Path to the temporary file to convert
        
    Returns:
        str: Extracted text from the document or None if conversion fails
    """
    output_path = temp_file_path + "_extracted"
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Convert document to text using LibreOffice
        cmd = [
            'libreoffice', '--headless', '--convert-to', 'txt:Text', 
            '--outdir', output_path, temp_file_path
        ]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if process.returncode != 0:
            error_msg = process.stderr.decode('utf-8', errors='ignore')
            st.error(f"LibreOffice conversion failed with error code {process.returncode}: {error_msg[:200]}...")
            return None
        
        # Get the output file (should be only one)
        converted_files = os.listdir(output_path)
        if not converted_files:
            st.error("LibreOffice conversion did not produce any output files")
            return None
        
        # Read the converted text file
        output_file = os.path.join(output_path, converted_files[0])
        with open(output_file, 'r', errors='ignore') as f:
            text = f.read()
            
        # Cleanup
        shutil.rmtree(output_path)
        return text
    except Exception as e:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        st.error(f"LibreOffice conversion error: {str(e)}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def process_document_cached(file_content, filename, file_extension):
    """
    Process document and extract text with content-based caching.
    
    Args:
        file_content: Binary content of the file
        filename: Name of the file
        file_extension: Extension of the file (pdf, txt, docx, etc.)
        
    Returns:
        str: Extracted text content or None if extraction fails
    """
    try:
        # Generate a content hash for caching
        content_hash = hashlib.md5(file_content).hexdigest()
        
        # Check if we have this document in our cache
        cache_key = f"doc_cache_{content_hash}"
        if cache_key in st.session_state:
            return st.session_state[cache_key]
            
        # Process based on file extension
        text = None
        
        if file_extension == "pdf":
            # Process PDF
            with io.BytesIO(file_content) as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                    
        elif file_extension == "txt":
            # Process plain text
            text = file_content.decode("utf-8", errors="replace")
            
        elif file_extension in ["docx"]:
            # Process DOCX
            with io.BytesIO(file_content) as docx_file:
                text = docx2txt.process(docx_file)
                
        elif file_extension == "csv":
            # Process CSV
            with io.BytesIO(file_content) as csv_file:
                csv_text = csv_file.read().decode("utf-8", errors="replace")
                reader = csv.reader(csv_text.splitlines())
                text = "\n".join([", ".join(row) for row in reader])
                
        elif file_extension in ["xlsx", "xls"]:
            # Process Excel
            with io.BytesIO(file_content) as excel_file:
                df = pd.read_excel(excel_file)
                text = df.to_string()
                
        elif file_extension == "json":
            # Process JSON
            with io.BytesIO(file_content) as json_file:
                json_text = json_file.read().decode("utf-8", errors="replace")
                # Parse and pretty print JSON for better text extraction
                parsed_json = json.loads(json_text)
                text = json.dumps(parsed_json, indent=2)
                
        elif file_extension in ["md", "markdown"]:
            # Process Markdown - just use the raw text
            text = file_content.decode("utf-8", errors="replace")
            
        elif file_extension in ["html", "htm"]:
            # Process HTML - extract text from tags
            with io.BytesIO(file_content) as html_file:
                html_text = html_file.read().decode("utf-8", errors="replace")
                soup = BeautifulSoup(html_text, "html.parser")
                text = soup.get_text(separator="\n")
                
        elif file_extension in ["tex", "ltx", "latex"]:
            # Process LaTeX
            text = process_latex_file(file_content)
            
        elif file_extension in LIBREOFFICE_EXTENSIONS:
            # Try LibreOffice conversion for other formats
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
                
            text = extract_text_with_libreoffice(temp_file_path)
            # Remove the temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
        # If all else fails, try the unstructured library
        if (text is None or text.strip() == "") and file_extension in UNSTRUCTURED_COMPATIBLE:
            try:
                with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                # Use unstructured to extract text
                elements = partition(filename=temp_file_path)
                text = "\n\n".join([str(element) for element in elements])
                
                # Remove the temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            except Exception as e:
                st.warning(f"Fallback extraction with unstructured failed: {str(e)}")
                
        # Cache the results
        if text:
            st.session_state[cache_key] = text
        else:
            st.error(f"Could not extract text from {filename}. The file may be empty, corrupted, or in an unsupported format.")
        
        return text
        
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return None

def process_latex_file(content):
    """
    Process LaTeX file content to extract plain text.
    
    Args:
        content (bytes): The LaTeX file content
        
    Returns:
        str: Plain text extracted from LaTeX
    """
    try:
        # Decode bytes to string, handling potential encoding issues
        try:
            latex_content = content.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            latex_content = content.decode('latin-1')
        
        # Convert LaTeX to plain text
        latex_converter = LatexNodes2Text()
        plain_text = latex_converter.latex_to_text(latex_content)
        
        # Handle math expressions (optional)
        # Find math expressions (e.g., $...$, $$...$$, \begin{equation}...\end{equation})
        math_expressions = re.findall(r'\$\$(.*?)\$\$|\$(.*?)\$|\\begin{equation}(.*?)\\end{equation}', 
                                     latex_content, re.DOTALL)
        
        # Add a description for each math expression found
        if math_expressions:
            plain_text += "\n\nMathematical expressions found in the document:\n"
            for expr in math_expressions:
                # Get the first non-empty group from the regex match tuple
                expr_text = next((x for x in expr if x), "")
                if expr_text.strip():
                    plain_text += f"- Mathematical expression: {expr_text}\n"
        
        return plain_text
    except Exception as e:
        st.error(f"Error processing LaTeX file: {str(e)}")
        return None

def process_document(file):
    """
    Extract text content from uploaded documents with performance tracking.
    
    Args:
        file: The uploaded file object
        
    Returns:
        str: Extracted text content or None if extraction fails
    """
    # Generate a file hash to check if already processed
    file_content = file.getvalue()
    file_hash = hashlib.md5(file_content).hexdigest()
    
    # If file is already in session state, use the cached result
    if file_hash in st.session_state.processed_files:
        return st.session_state.processed_files[file_hash]
    
    # Get file extension
    file_extension = file.name.split('.')[-1].lower()
    
    # Measure processing time
    start_time = time.time()
    
    # Process using cached function
    text = process_document_cached(file_content, file.name, file_extension)
    
    # Update performance metrics
    st.session_state.performance_metrics["document_processing_time"] += time.time() - start_time
    
    # Store in session state for future reference
    if text:
        st.session_state.processed_files[file_hash] = text
    
    return text

@st.cache_data(show_spinner=False)
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into chunks with overlap for better context preservation.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters
        
    Returns:
        list: List of text chunks as Document objects
    """
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.create_documents([text])
    except Exception as e:
        st.error(f"Error chunking text: {str(e)}")
        # Return a single document with the text to prevent processing failures
        return [Document(page_content=text)]

def ingest_documents(files):
    """
    Process documents, create chunks, and store in FAISS vector index.
    
    Processing workflow:
    1. Extract text from each document
    2. Split text into chunks with overlap
    3. Create vector embeddings for each chunk
    4. Store vectors in FAISS index for efficient similarity search
    5. Save FAISS index to disk for persistence
    
    Args:
        files: List of uploaded file objects
        
    Returns:
        bool: True if ingestion was successful, False otherwise
    """
    all_chunks = []
    texts = []
    success_count = 0
    failed_count = 0
    
    # Create a progress bar for document processing
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        # Update progress
        progress_percentage = int((i / len(files)) * 100)
        progress_bar.progress(progress_percentage)
        
        with st.status(f"Processing {file.name}... ({i+1}/{len(files)})"):
            # Extract text from each document
            text = process_document(file)
            if text and len(text.strip()) > 0:
                # Save the original file for reference - using a sanitized filename
                try:
                    # Sanitize filename to avoid issues with special characters and length
                    filename = os.path.basename(file.name)
                    # Limit filename length to 100 characters to avoid path length issues
                    if len(filename) > 100:
                        base, ext = os.path.splitext(filename)
                        filename = base[:96-len(ext)] + ext
                    
                    # Ensure the documents directory exists
                    os.makedirs(DOCS_DIR, exist_ok=True)
                    
                    # Save the file with the sanitized name
                    safe_path = os.path.join(DOCS_DIR, filename)
                    with open(safe_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    texts.append(text)
                    success_count += 1
                    
                    # Generate chunks for this document
                    start_time = time.time()
                    chunks = chunk_text(text)
                    all_chunks.extend(chunks)
                    st.session_state.performance_metrics["embedding_generation_time"] += time.time() - start_time
                    
                    st.success(f"✅ Successfully processed {file.name} - Created {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"Error saving file {file.name}: {str(e)}")
                    failed_count += 1
            else:
                failed_count += 1
                st.error(f"❌ Failed to extract text from {file.name}")
    
    # Complete the progress bar
    progress_bar.progress(100)
    
    if success_count > 0:
        st.info(f"Successfully processed {success_count} documents. Failed to process {failed_count} documents.")
    
    # Only proceed if we have valid chunks
    if all_chunks:
        with st.status("Creating FAISS vector index..."):
            # Measure embedding time
            start_time = time.time()
            
            # Create or update the vector store with document chunks
            vector_store = FAISS.from_documents(
                all_chunks, 
                embeddings
            )
            
            # Save the FAISS index using binary format instead of pickle
            index_path = os.path.join(FAISS_INDEX_DIR, "index.bin")
            docstore_path = os.path.join(FAISS_INDEX_DIR, "docstore.pkl")
            
            # Save FAISS index in binary format
            faiss.write_index(vector_store.index, index_path)
            
            # We still need to save the docstore which contains the mapping from IDs to documents
            # This is separate from the index and doesn't pose the same security risks
            with open(docstore_path, 'wb') as f:
                pickle.dump(vector_store.docstore, f)
                
            # Store in session state
            st.session_state.vector_store = vector_store
            
            elapsed_time = time.time() - start_time
            st.session_state.performance_metrics["embedding_generation_time"] += elapsed_time
            
            st.success(f"Created FAISS index with {len(all_chunks)} chunks from {len(texts)} documents in {elapsed_time:.2f} seconds")
            return True
    return False

def clear_vector_store():
    """
    Clear the FAISS vector index and reset the session state.
    
    When using Docker volumes, we can't directly access the underlying files
    but we can recreate an empty FAISS index and clear all session data.
    
    This function:
    1. Clears the vector store from session state
    2. Creates a new empty FAISS index
    3. Clears all document files from the documents directory
    4. Clears all caches to ensure fresh data
    5. Clears uploaded files from the session state
    
    Returns:
        bool: True if the operation was successful
    """
    global embeddings
    try:
        # Clear the session state
        st.session_state.vector_store = None
        st.session_state.processed_files = {}
        st.session_state.performance_metrics = {
            "document_processing_time": 0,
            "embedding_generation_time": 0,
            "query_time": 0
        }
        
        # Clear the file uploader by removing any 'uploadedFiles' key from session state
        if 'uploadedFiles' in st.session_state:
            del st.session_state['uploadedFiles']
        
        # Create empty directories if they don't exist
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        os.makedirs(DOCS_DIR, exist_ok=True)
        
        # Clear all files from the documents directory
        for filename in os.listdir(DOCS_DIR):
            file_path = os.path.join(DOCS_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    st.warning(f"Failed to remove file {filename}: {str(e)}")
        
        # Clear all files from FAISS directory
        for filename in os.listdir(FAISS_INDEX_DIR):
            file_path = os.path.join(FAISS_INDEX_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    st.warning(f"Failed to remove file {filename}: {str(e)}")
        
        # In Docker volume mode, we'll create a new empty FAISS index
        empty_docs = [Document(page_content="Empty document")]
        empty_vector_store = FAISS.from_documents(
            empty_docs, 
            embeddings
        )
        
        # Save empty index in binary format
        index_path = os.path.join(FAISS_INDEX_DIR, "index.bin")
        docstore_path = os.path.join(FAISS_INDEX_DIR, "docstore.pkl")
        
        faiss.write_index(empty_vector_store.index, index_path)
        with open(docstore_path, 'wb') as f:
            pickle.dump(empty_vector_store.docstore, f)
        
        st.info("FAISS index and document files have been reset")
        
        # Clear all caches to make sure cached data is refreshed
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Reload embedding model after clearing cache
        embeddings = get_embeddings_model()
        
        return True
    except Exception as e:
        st.error(f"Error while clearing vector store: {str(e)}")
        return False

@st.cache_resource(show_spinner="Loading FAISS index...")
def get_vector_store():
    """
    Load and cache the FAISS vector store from disk.
    
    This function checks if a FAISS index exists and loads it
    using the current embedding model.
    
    Returns:
        FAISS: The loaded vector store or None if no index exists
    """
    # Check if FAISS index exists
    index_path = os.path.join(FAISS_INDEX_DIR, "index.bin")
    docstore_path = os.path.join(FAISS_INDEX_DIR, "docstore.pkl")
    
    if os.path.exists(index_path) and os.path.exists(docstore_path):
        try:
            # Load FAISS index from binary format
            index = faiss.read_index(index_path)
            
            # Load docstore
            with open(docstore_path, 'rb') as f:
                docstore = pickle.load(f)
            
            # Create the FAISS vector store with loaded components
            vector_store = FAISS(embeddings.embed_query, index, docstore, {})
            return vector_store
        except Exception as e:
            st.error(f"Error loading FAISS index: {str(e)}")
            return None
    return None

@st.cache_resource(show_spinner="Connecting to LLM...")
def get_llm():
    """
    Initialize and cache the LLM connection.
    
    Returns:
        Ollama: Configured LLM instance
    """
    return Ollama(
        model="llama3:8b",  # Using Llama 3 8B model for balanced performance and efficiency
        base_url=OLLAMA_BASE_URL,  # Connect to local Ollama installation
        temperature=0.1  # Lower temperature for more focused responses
    )

@st.cache_resource(show_spinner="Preparing QA chain...")
def get_qa_chain(_llm):
    """
    Create and cache the question-answering chain.

    Args:
        _llm: The language model
        
    Returns:
        Chain: The configured QA chain
    """
    return load_qa_chain(_llm, chain_type="stuff", 
                       prompt=PromptTemplate(
                           template="Context:\n{context}\nQuestion: {question}\nAnswer:",
                           input_variables=["context", "question"]
                       ))

@st.cache_data(ttl=600, show_spinner=False)  # Cache for 10 minutes
def cached_similarity_search(_vector_store, query, k=3):
    """
    Perform FAISS similarity search with caching.
    
    This function finds the k-nearest neighbors to the query vector
    in the FAISS vector space.
    
    Args:
        _vector_store: The FAISS vector store to search
        query: The search query
        k: Number of results to return
        
    Returns:
        list: Relevant documents
    """
    return _vector_store.similarity_search(query, k=k)

@st.cache_data(ttl=300, show_spinner=False)
def cached_answer_generation(query_hash, _docs_content, question):
    """
    Generate an answer using retrieved documents with caching.
    
    Args:
        query_hash: Hash of the query for cache key
        _docs_content: Content of retrieved documents
        question: The question to answer
        
    Returns:
        dict: Contains the generated answer
    """
    llm = get_llm()
    chain = get_qa_chain(llm)
    return chain({"input_documents": _docs_content, "question": question}, return_only_outputs=True)

def query_documents(query):
    """
    Retrieve relevant document chunks using FAISS and generate an answer.
    
    This function:
    1. Converts the query to a vector
    2. Uses FAISS to find similar document chunks
    3. Passes retrieved chunks to the LLM
    4. Generates a specific answer
    
    Args:
        query: The user's question
        
    Returns:
        dict: Contains the generated answer and other information
    """
    # First check if LLM is ready
    if not check_llm_status():
        st.warning("⚠️ Ollama is not running or the model is not available. Some features will not work.")
        return None
        
    # Measure query time
    start_time = time.time()
    
    # Load vector store if not already in session state
    if not st.session_state.vector_store:
        st.session_state.vector_store = get_vector_store()
        if not st.session_state.vector_store:
            st.error("No documents have been ingested yet!")
            return None
    
    try:
        # Show progress of the query process
        with st.status("Processing your question...", expanded=True) as status:
            status.update(label="🔍 Finding relevant documents using FAISS...")
            # Retrieve relevant documents using FAISS similarity search
            docs = cached_similarity_search(st.session_state.vector_store, query, k=3)
            
            # If no relevant documents are found
            if not docs:
                st.warning("No relevant documents found for this query.")
                return None
            
            status.update(label="🧠 Generating answer with Ollama...")
            # Create hash for the query and docs
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            # Generate response using cached function
            result = cached_answer_generation(query_hash, docs, query)
            
            status.update(label="✅ Answer generation complete!", state="complete")
        
        # Update performance metrics
        st.session_state.performance_metrics["query_time"] += time.time() - start_time
        
        return result
    except Exception as e:
        st.error(f"Error during query processing: {str(e)}")
        # Provide additional debugging information
        st.info("If you're seeing errors, try clearing the FAISS index and reprocessing your documents.")
        return None

def display_performance_metrics():
    """
    Display performance metrics in a formatted way.
    Shows document processing time, embedding generation time, and query time.
    """
    metrics = st.session_state.performance_metrics
    
    st.markdown("### Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Document Processing", f"{metrics['document_processing_time']:.2f}s")
    
    with col2:
        st.metric("Embedding Generation", f"{metrics['embedding_generation_time']:.2f}s")
    
    with col3:
        st.metric("Query Time", f"{metrics['query_time']:.2f}s")

###########################################
# STREAMLIT UI
###########################################

# App title and structure
st.title("Minimal RAG Pipeline with FAISS")
st.markdown("A Retrieval-Augmented Generation system using FAISS vector search for efficient document retrieval and Ollama for local LLM inference.")

# Create tabs for document upload and querying
tab1, tab2, tab3, tab4 = st.tabs(["Upload Documents", "Ask Questions", "Performance", "System Status"])

# Check if LLM is loaded
llm_ready = check_llm_status()
if not llm_ready:
    st.warning(f"⚠️ {REQUIRED_MODEL} model is not available. Some features will not work.")
    # Add a loading animation
    with st.spinner(f"Checking {REQUIRED_MODEL} model status..."):
        # Check status periodically (but don't block for too long)
        for _ in range(3):  # Just check a few times to avoid blocking UI
            time.sleep(1)
            if check_llm_status():
                st.success(f"✅ {REQUIRED_MODEL} model loaded successfully!")
                llm_ready = True
                break
else:
    st.success(f"✅ {REQUIRED_MODEL} model is loaded and ready to use!")

# System Status Tab
with tab4:
    st.header("System Status")
    
    st.subheader("Ollama Status")
    display_ollama_status()
    
    st.subheader("Environment Information")
    st.json({
        "Python Version": sys.version.split()[0],
        "Ollama Host": OLLAMA_HOST,
        "Ollama URL": OLLAMA_BASE_URL,
        "Required Model": REQUIRED_MODEL,
        "Application Mode": "Hybrid (containerized app with host Ollama)",
        "FAISS Index Directory": FAISS_INDEX_DIR,
        "Documents Directory": DOCS_DIR
    })
    
    if st.button("Refresh Status"):
        st.experimental_rerun()

# Document Upload Tab
with tab1:
    st.header("Upload Documents")
    
    st.markdown("""
    ### Supported Document Formats:
    - PDF (`.pdf`)
    - Word Documents (`.docx`, `.doc`)
    - Text Files (`.txt`)
    - Excel Spreadsheets (`.xlsx`, `.xls`)
    - CSV Files (`.csv`)
    - JSON Files (`.json`)
    - Markdown (`.md`)
    - HTML Files (`.html`, `.htm`)
    - PowerPoint (`.pptx`, `.ppt`)
    - LaTeX Documents (`.tex`, `.ltx`, `.latex`)
    - And more...
    """)
    
    # Add document troubleshooting expander
    with st.expander("🔧 Document Troubleshooting Tips"):
        st.markdown("""
        ### Common Issues and Solutions
        
        **Legacy .doc files fail to process:**
        - Try saving the document as .docx format in Microsoft Word
        - Use LibreOffice to convert .doc to .docx
        - Some very old .doc files may require manual conversion
        
        **PDF text extraction issues:**
        - Ensure the PDF contains actual text (not just scanned images)
        - For scanned PDFs, use OCR software first
        
        **Large files process slowly:**
        - Consider splitting large documents into smaller parts
        - Simplify complex formatting in source documents
        
        **Document conversion failed with all methods:**
        - Try opening and re-saving the document in a different application
        - Check if the document is password-protected or corrupted
        - Convert to a simpler format like plain text (.txt) if content is what matters
        """)
    
    # Add FAISS vector search expander
    with st.expander("🔍 How Vector Search Works"):
        st.markdown("""
        ### FAISS Vector Search Technology
        
        This application uses FAISS (Facebook AI Similarity Search) to power its document search capabilities:
        
        **How It Works:**
        1. **Document Processing**: Your uploaded documents are processed and split into text chunks
        2. **Vector Embedding**: Each chunk is converted to a numerical vector using a neural network
        3. **FAISS Indexing**: These vectors are stored in a specialized FAISS index for efficient retrieval
        4. **Similarity Search**: When you ask a question, it's converted to a vector and compared to all stored vectors
        5. **Retrieval**: The most similar document chunks are retrieved and sent to the LLM
        6. **Answer Generation**: The LLM uses these relevant chunks to generate a specific answer
        
        **Advantages of FAISS:**
        - High performance similarity search
        - Efficient memory usage
        - Scalable to millions of documents
        - Persists between application restarts
        
        All vectors are stored locally in the `./data/faiss` directory.
        """)
    
    files = st.file_uploader("Upload documents", 
                             type=list(SUPPORTED_EXTENSIONS.keys()), 
                             accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Process Documents"):
            if files:
                with st.spinner("Processing documents..."):
                    if ingest_documents(files):
                        st.success("All documents have been processed and indexed in FAISS!")
            else:
                st.warning("Please upload files first.")
    
    with col2:
        if st.button("Clear Vector Store & Uploads", type="secondary"):
            with st.spinner("Clearing FAISS index, uploads, and all caches..."):
                if clear_vector_store():
                    st.success("✅ FAISS index has been cleared. All uploaded documents and caches have been reset.")
    
    # Add a note about the clear button functionality
    st.info("💡 Use the 'Clear Vector Store & Uploads' button to remove all uploaded documents, indexed documents, and clear all caches.")
    
    # Show already processed documents
    if os.path.exists(DOCS_DIR):
        doc_files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
        if doc_files:
            st.subheader("Processed Documents")
            for doc in doc_files:
                ext = doc.split('.')[-1].lower()
                doc_type = SUPPORTED_EXTENSIONS.get(ext, "Unknown Format")
                st.markdown(f"📄 **{doc}** - *{doc_type}*")

# Query Tab
with tab2:
    st.header("Ask Questions")
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                start_time = time.time()
                result = query_documents(query)
                query_time = time.time() - start_time
                
                if result:
                    st.subheader("Answer:")
                    st.write(result["output_text"])
                    st.info(f"⚡ Answer generated in {query_time:.2f} seconds")

# Performance Tab
with tab3:
    st.header("Performance Metrics")
    display_performance_metrics()
    
    if st.button("Reset Metrics"):
        st.session_state.performance_metrics = {
            "document_processing_time": 0,
            "embedding_generation_time": 0,
            "query_time": 0
        }
        st.success("Performance metrics have been reset.")
    
    st.markdown("""
    ### FAISS Vector Search
    
    This application uses FAISS (Facebook AI Similarity Search) for vector storage and retrieval:
    
    - **High Performance**: FAISS is designed for efficient similarity search in high-dimensional spaces
    - **Memory Efficient**: Optimized for both memory usage and search speed
    - **Scalable**: Can handle millions of vectors with minimal performance degradation
    - **Persistence**: FAISS indices are saved to disk and loaded on demand
    
    ### Caching Information
    
    This RAG pipeline uses multiple levels of caching to improve performance:
    
    1. **Document Processing Cache**: Prevents reprocessing the same document (TTL: 1 hour)
    2. **Text Chunking Cache**: Avoids rechunking the same text
    3. **FAISS Index Cache**: Keeps the vector store in memory between queries
    4. **LLM Connection Cache**: Maintains the connection to the LLM
    5. **Similarity Search Cache**: Caches search results for identical queries (TTL: 10 minutes)
    6. **Answer Generation Cache**: Caches answers to identical questions (TTL: 5 minutes)
    
    All caches are automatically cleared when you click "Clear Vector Store & Uploads".
    """) 