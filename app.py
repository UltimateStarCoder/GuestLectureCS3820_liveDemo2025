import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
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

###########################################
# INITIALIZATION AND CONFIGURATION
###########################################

# Setup directories for storing data
CHROMA_DIR, DOCS_DIR = "./data/chroma", "./documents"
os.makedirs(CHROMA_DIR, exist_ok=True)  # Create directory for vector database
os.makedirs(DOCS_DIR, exist_ok=True)    # Create directory for document storage

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

# Cache the embeddings model to avoid reloading it on every rerun
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings_model():
    """
    Load and cache the sentence embedding model.
    This prevents reloading the model on every rerun, which is expensive.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# We use a small but effective model from Hugging Face
embeddings = get_embeddings_model()

# Define supported file extensions
SUPPORTED_FORMATS = {
    # Document formats
    'pdf': 'PDF Document',
    'txt': 'Text File',
    'md': 'Markdown File',
    'doc': 'Word Document (Legacy)',
    'docx': 'Word Document',
    'rtf': 'Rich Text Format',
    'odt': 'OpenDocument Text',
    
    # Presentation formats
    'ppt': 'PowerPoint Presentation (Legacy)',
    'pptx': 'PowerPoint Presentation',
    'odp': 'OpenDocument Presentation',
    
    # Spreadsheet formats
    'csv': 'CSV File',
    'xlsx': 'Excel Spreadsheet',
    'xls': 'Excel Spreadsheet (Legacy)',
    'ods': 'OpenDocument Spreadsheet',
    
    # Data formats
    'json': 'JSON File',
    'xml': 'XML File',
    
    # Web formats
    'html': 'HTML File',
    'htm': 'HTML File',
}

###########################################
# DOCUMENT PROCESSING FUNCTIONS
###########################################

@st.cache_data(show_spinner=False)
def extract_text_with_libreoffice(temp_file_path):
    """
    Use LibreOffice to convert documents to text format.
    Useful for handling various office document formats.
    This function is cached to avoid repeated conversions.
    
    Args:
        temp_file_path: Path to the temporary file to convert
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

# Cache document processing results to avoid reprocessing the same document
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def process_document_cached(file_content, filename, file_extension):
    """
    Cached version of document processing.
    Processes the document and returns the extracted text.
    Results are cached based on file content hash.
    """
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    text = None
    try:
        # Process based on file type
        if file_extension == 'pdf':
            # Process PDF files
            pdf = PdfReader(temp_file_path)
            text = "".join([page.extract_text() for page in pdf.pages])
            
        elif file_extension == 'txt':
            # Process text files
            text = file_content.decode("utf-8")
            
        elif file_extension in ['doc', 'docx']:
            # First try with docx2txt for .docx format
            try:
                text = docx2txt.process(temp_file_path)
            except Exception as docx_error:
                # If that fails (especially for older .doc files), fallback to LibreOffice
                st.info(f"Trying alternative conversion method for {filename}...")
                text = extract_text_with_libreoffice(temp_file_path)
                
                # If LibreOffice fails too, try with unstructured as last resort
                if not text:
                    try:
                        elements = partition(temp_file_path)
                        text = "\n\n".join([str(element) for element in elements])
                    except Exception as e:
                        st.error(f"All conversion methods failed for {filename}. Please try converting the file to .docx format.")
                        text = None
            
        elif file_extension == 'csv':
            # Process CSV files
            df = pd.read_csv(temp_file_path)
            text = df.to_string()
            
        elif file_extension in ['xls', 'xlsx']:
            # Process Excel files
            df = pd.read_excel(temp_file_path)
            text = df.to_string()
            
        elif file_extension == 'json':
            # Process JSON files
            with open(temp_file_path, 'r') as f:
                data = json.load(f)
            text = json.dumps(data, indent=2)
            
        elif file_extension in ['md', 'markdown']:
            # Process Markdown files
            with open(temp_file_path, 'r') as f:
                text = f.read()
                
        elif file_extension in ['html', 'htm']:
            # Process HTML files
            with open(temp_file_path, 'r') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                text = soup.get_text(separator='\n')
                
        elif file_extension in ['rtf', 'odt', 'ppt', 'pptx', 'odp', 'ods']:
            # Use LibreOffice for these formats
            text = extract_text_with_libreoffice(temp_file_path)
            
        else:
            # Try with unstructured for unknown types
            try:
                elements = partition(temp_file_path)
                text = "\n\n".join([str(element) for element in elements])
            except Exception as e:
                st.error(f"Unstructured processing error for {filename}: {str(e)}")
                text = None
    
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        text = None
    
    # Clean up the text
    if text:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    
    # Remove the temporary file
    try:
        os.unlink(temp_file_path)
    except:
        pass
    
    return text

def process_document(file):
    """
    Extract text content from uploaded documents.
    Uses a cached implementation for better performance.
    
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

# Cache text chunking for better performance
@st.cache_data(show_spinner=False)
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into chunks using a cached function.
    This avoids re-chunking the same text multiple times.
    
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
        from langchain.schema import Document
        return [Document(page_content=text)]

def ingest_documents(files):
    """
    Process multiple documents, create chunks, and store in vector database.
    This is the document ingestion phase of RAG.
    
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
        with st.status("Creating vector embeddings..."):
            # Measure embedding time
            start_time = time.time()
            
            # Create or update the vector store with document chunks
            # This converts text chunks to vector embeddings and stores them
            vector_store = Chroma.from_documents(all_chunks, embeddings, persist_directory=CHROMA_DIR)
            vector_store.persist()  # Save to disk
            st.session_state.vector_store = vector_store
            
            elapsed_time = time.time() - start_time
            st.session_state.performance_metrics["embedding_generation_time"] += elapsed_time
            
            st.success(f"Created {len(all_chunks)} chunks from {len(texts)} documents in {elapsed_time:.2f} seconds")
            return True
    return False

def clear_vector_store():
    """
    Clear the vector storage and reset the session state.
    This allows the user to start fresh without previously indexed documents.
    Also clears all caches to ensure fresh data.
    
    Returns:
        bool: True if the operation was successful
    """
    try:
        # Clear the session state
        st.session_state.vector_store = None
        st.session_state.processed_files = {}
        st.session_state.performance_metrics = {
            "document_processing_time": 0,
            "embedding_generation_time": 0,
            "query_time": 0
        }
        
        # If the ChromaDB directory exists, remove all its contents
        if os.path.exists(CHROMA_DIR):
            # List files before deletion for verification
            existing_files = []
            for root, dirs, files in os.walk(CHROMA_DIR):
                for file in files:
                    existing_files.append(os.path.join(root, file))
            
            # Delete the directory contents
            shutil.rmtree(CHROMA_DIR)
            os.makedirs(CHROMA_DIR)  # Recreate the empty directory
            
            st.info(f"Removed {len(existing_files)} files from vector store")
        
        # Clear the documents directory
        doc_count = 0
        if os.path.exists(DOCS_DIR):
            for file in os.listdir(DOCS_DIR):
                file_path = os.path.join(DOCS_DIR, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    doc_count += 1
            
            st.info(f"Removed {doc_count} document files")
        
        # Force sync to disk
        os.sync()
        
        # Clear all caches to make sure cached data is refreshed
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Reload embedding model after clearing cache
        global embeddings
        embeddings = get_embeddings_model()
        
        return True
    except Exception as e:
        st.error(f"Error while clearing vector store: {str(e)}")
        return False

# Cache vector store loading to avoid reloading on each query
@st.cache_resource(show_spinner="Loading vector store...")
def get_vector_store(chroma_dir, _embeddings_function):
    """
    Load and cache the vector store.
    This prevents reloading the store on every query, which is expensive.
    
    Args:
        chroma_dir: Directory where ChromaDB data is stored
        _embeddings_function: The embedding function (not used for caching)
    """
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        return Chroma(
            persist_directory=chroma_dir, 
            embedding_function=_embeddings_function
        )
    return None

# Cache the LLM to avoid reinitializing it for each query
@st.cache_resource(show_spinner="Connecting to LLM...")
def get_llm():
    """
    Initialize and cache the LLM connection.
    This prevents reconnecting to the LLM service on every query.
    """
    return Ollama(
        model="tinyllama",  # Using TinyLlama model (1.1B parameters) for edge devices
        base_url="http://ollama:11434",  # Connect to Ollama service in Docker
        temperature=0.1  # Lower temperature for more focused responses
    )

# Cache question-answering chains for better performance
@st.cache_resource(show_spinner="Preparing QA chain...")
def get_qa_chain(_llm):
    """
    Create and cache the question-answering chain.
    The chain combines the LLM with a prompt template.
    
    Args:
        _llm: The language model (not used for caching)
    """
    return load_qa_chain(_llm, chain_type="stuff", 
                       prompt=PromptTemplate(
                           template="Context:\n{context}\nQuestion: {question}\nAnswer:",
                           input_variables=["context", "question"]
                       ))

# Cache similarity search to avoid repeated searches for the same query
@st.cache_data(ttl=600, show_spinner=False)  # Cache for 10 minutes
def cached_similarity_search(_vector_store, query, k=3):
    """
    Perform a cached similarity search.
    This prevents repeating the same search for identical queries.
    
    Args:
        _vector_store: The vector store to search in (not used for caching)
        query: The query to search for
        k: Number of results to return
        
    Returns:
        list: Relevant documents
    """
    return _vector_store.similarity_search(query, k=k)

@st.cache_data(ttl=300, show_spinner=False)
def cached_answer_generation(query_hash, _docs_content, question):
    """
    Generate an answer for a query with cached results.
    Caches answers for 5 minutes to avoid redundant LLM calls.
    
    Args:
        query_hash: The hash of the query for cache key
        _docs_content: The content of the documents (not used for caching)
        question: The actual question to answer
    """
    llm = get_llm()
    chain = get_qa_chain(llm)
    return chain({"input_documents": _docs_content, "question": question}, return_only_outputs=True)

def query_documents(query):
    """
    Retrieve relevant document chunks and generate an answer using the LLM.
    This implements the retrieval and generation phases of RAG.
    Uses caching for better performance.
    
    Args:
        query: The user's question
        
    Returns:
        dict: Contains the generated answer and other information
    """
    # Measure query time
    start_time = time.time()
    
    # Load vector store if not already in session state
    if not st.session_state.vector_store:
        st.session_state.vector_store = get_vector_store(CHROMA_DIR, embeddings)
        if not st.session_state.vector_store:
            st.error("No documents have been ingested yet!")
            return None
    
    try:
        # Retrieve relevant documents using semantic similarity (with caching)
        # This is the retrieval phase of RAG
        docs = cached_similarity_search(st.session_state.vector_store, query, k=3)
        
        # If no relevant documents are found
        if not docs:
            st.warning("No relevant documents found for this query.")
            return None
        
        # Create hash for the query and docs
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Generate response using cached function to avoid repeated LLM calls
        result = cached_answer_generation(query_hash, docs, query)
        
        # Update performance metrics
        st.session_state.performance_metrics["query_time"] += time.time() - start_time
        
        return result
    except Exception as e:
        st.error(f"Error during query processing: {str(e)}")
        # Provide additional debugging information
        st.info("If you're seeing caching errors, try clearing the vector store and reprocessing your documents.")
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
st.title("Minimal RAG Pipeline")

# Create tabs for document upload and querying
tab1, tab2, tab3 = st.tabs(["Upload Documents", "Ask Questions", "Performance"])

# Document Upload Tab
with tab1:
    st.header("Upload Documents")
    
    st.markdown("""
    ### Supported Document Formats:
    - Text Files (txt, md)
    - PDF Documents (pdf)
    - Microsoft Office (doc, docx, ppt, pptx, xls, xlsx)
    - OpenDocument Formats (odt, odp, ods)
    - Data Files (csv, json, xml)
    - Web Files (html, htm)
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
    
    files = st.file_uploader("Upload documents", 
                             type=list(SUPPORTED_FORMATS.keys()), 
                             accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Process Documents"):
            if files:
                with st.spinner("Processing documents..."):
                    if ingest_documents(files):
                        st.success("All documents have been processed and indexed!")
            else:
                st.warning("Please upload files first.")
    
    with col2:
        if st.button("Clear Vector Store", type="secondary"):
            with st.spinner("Clearing vector store and all caches..."):
                if clear_vector_store():
                    st.success("✅ Vector store has been cleared. All indexed documents and caches have been reset.")
    
    # Add a note about the clear button functionality
    st.info("💡 Use the 'Clear Vector Store' button to remove all indexed documents and clear all caches.")
    
    # Show already processed documents
    if os.path.exists(DOCS_DIR):
        doc_files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
        if doc_files:
            st.subheader("Processed Documents")
            for doc in doc_files:
                ext = doc.split('.')[-1].lower()
                doc_type = SUPPORTED_FORMATS.get(ext, "Unknown Format")
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
    ### Caching Information
    
    This RAG pipeline uses multiple levels of caching to improve performance:
    
    1. **Document Processing Cache**: Prevents reprocessing the same document (TTL: 1 hour)
    2. **Text Chunking Cache**: Avoids rechunking the same text
    3. **Vector Store Cache**: Keeps the vector store in memory between queries
    4. **LLM Connection Cache**: Maintains the connection to the LLM
    5. **Similarity Search Cache**: Caches search results for identical queries (TTL: 10 minutes)
    6. **Answer Generation Cache**: Caches answers to identical questions (TTL: 5 minutes)
    
    All caches are automatically cleared when you click "Clear Vector Store".
    """) 