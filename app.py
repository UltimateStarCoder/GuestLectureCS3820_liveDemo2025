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
from unstructured.partition.auto import partition

###########################################
# INITIALIZATION AND CONFIGURATION
###########################################

# Setup directories for storing data
CHROMA_DIR, DOCS_DIR = "./data/chroma", "./documents"
os.makedirs(CHROMA_DIR, exist_ok=True)  # Create directory for vector database
os.makedirs(DOCS_DIR, exist_ok=True)    # Create directory for document storage

# Initialize Streamlit session state to persist the vector store between reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Create embeddings model that converts text to vector representations
# We use a small but effective model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

def extract_text_with_libreoffice(temp_file_path):
    """
    Use LibreOffice to convert documents to text format.
    Useful for handling various office document formats.
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
            return None
        
        # Get the output file (should be only one)
        converted_files = os.listdir(output_path)
        if not converted_files:
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

def process_document(file):
    """
    Extract text content from uploaded documents.
    Supports a wide variety of document formats.
    
    Args:
        file: The uploaded file object
        
    Returns:
        str: Extracted text content or None if extraction fails
    """
    # Get file extension
    file_extension = file.name.split('.')[-1].lower()
    
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
        temp_file.write(file.getvalue())
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
            text = file.getvalue().decode("utf-8")
            
        elif file_extension in ['doc', 'docx']:
            # Process Word documents
            text = docx2txt.process(temp_file_path)
            
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
                st.error(f"Unstructured processing error for {file.name}: {str(e)}")
                text = None
    
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
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

def ingest_documents(files):
    """
    Process multiple documents, create chunks, and store in vector database.
    This is the document ingestion phase of RAG.
    
    Args:
        files: List of uploaded file objects
        
    Returns:
        bool: True if ingestion was successful, False otherwise
    """
    texts = []
    success_count = 0
    failed_count = 0
    
    for file in files:
        with st.status(f"Processing {file.name}..."):
            # Extract text from each document
            text = process_document(file)
            if text and len(text.strip()) > 0:
                # Save the original file for reference
                with open(os.path.join(DOCS_DIR, file.name), "wb") as f:
                    f.write(file.getvalue())
                texts.append(text)
                success_count += 1
                st.success(f"✅ Successfully processed {file.name}")
            else:
                failed_count += 1
                st.error(f"❌ Failed to extract text from {file.name}")
    
    if success_count > 0:
        st.info(f"Successfully processed {success_count} documents. Failed to process {failed_count} documents.")
    
    # Only proceed if we have valid text content
    if texts:
        with st.status("Creating vector embeddings..."):
            # Split text into smaller chunks for better retrieval
            # This is a crucial step in RAG to enable precise context retrieval
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents(texts)
            
            # Create or update the vector store with document chunks
            # This converts text chunks to vector embeddings and stores them
            vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
            vector_store.persist()  # Save to disk
            st.session_state.vector_store = vector_store
            
            st.success(f"Created {len(chunks)} chunks from {len(texts)} documents")
            return True
    return False

###########################################
# QUERY AND RESPONSE GENERATION
###########################################

def query_documents(query):
    """
    Retrieve relevant document chunks and generate an answer using the LLM.
    This implements the retrieval and generation phases of RAG.
    
    Args:
        query: The user's question
        
    Returns:
        dict: Contains the generated answer and other information
    """
    # Load vector store if not already in session state
    if not st.session_state.vector_store:
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            st.session_state.vector_store = Chroma(
                persist_directory=CHROMA_DIR, 
                embedding_function=embeddings
            )
        else:
            st.error("No documents have been ingested yet!")
            return None
    
    # Retrieve relevant documents using semantic similarity
    # This is the retrieval phase of RAG
    docs = st.session_state.vector_store.similarity_search(query, k=3)
    
    # Generate response using Ollama LLM
    # This is the generation phase of RAG
    llm = Ollama(
        model="tinyllama",  # Using TinyLlama model (1.1B parameters) for edge devices
        base_url="http://ollama:11434",  # Connect to Ollama service in Docker
        temperature=0.1  # Lower temperature for more focused responses
    )
    
    # Create a question-answering chain that combines retrieved docs with the query
    # The prompt template instructs the model how to use the context
    chain = load_qa_chain(llm, chain_type="stuff", 
                         prompt=PromptTemplate(
                             template="Context:\n{context}\nQuestion: {question}\nAnswer:",
                             input_variables=["context", "question"]
                         ))
    
    # Run the chain to get a response
    return chain({"input_documents": docs, "question": query}, return_only_outputs=True)

###########################################
# STREAMLIT UI
###########################################

# App title and structure
st.title("Minimal RAG Pipeline")

# Create tabs for document upload and querying
tab1, tab2 = st.tabs(["Upload Documents", "Ask Questions"])

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
    
    files = st.file_uploader("Upload documents", 
                             type=list(SUPPORTED_FORMATS.keys()), 
                             accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if files:
            with st.spinner("Processing documents..."):
                if ingest_documents(files):
                    st.success("All documents have been processed and indexed!")
        else:
            st.warning("Please upload files first.")
    
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
            with st.spinner("Generating answer... (This may take a moment when first using a model)"):
                result = query_documents(query)
                if result:
                    st.subheader("Answer:")
                    st.write(result["output_text"]) 