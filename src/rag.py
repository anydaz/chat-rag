"""RAG module for document retrieval with Chroma."""

import os
import chromadb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader


def load_pdfs_from_directory(pdf_dir: str = "data/pdfs") -> str:
    """Load all PDFs from a directory and combine text."""
    pdf_path = Path(pdf_dir)
    
    print(f"Looking for PDFs in: {pdf_path.absolute()}")
    
    if not pdf_path.exists():
        print(f"❌ Directory does not exist: {pdf_path}")
        return ""
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    combined_text = ""
    for pdf_file in pdf_files:
        try:
            print(f"  Reading: {pdf_file.name}")
            reader = PdfReader(pdf_file)
            num_pages = len(reader.pages)
            print(f"    Pages: {num_pages}")
            
            for page in reader.pages:
                combined_text += page.extract_text() + "\n"
        except Exception as e:
            print(f"  ❌ Error reading {pdf_file}: {e}")
    
    return combined_text


def initialize_chroma():
    """Initialize Chroma vector store with OpenAI embeddings."""
    # Create persistent Chroma client
    client = chromadb.PersistentClient(path="./data/chroma")
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    return client, embeddings


def load_documents_to_chroma(pdf_dir: str = "data/pdfs"):
    """Load PDFs and store embeddings in Chroma. Checks for new PDFs each time."""
    print(f"\n=== Loading Documents to Chroma ===")
    print(f"PDF Directory: {pdf_dir}")
    
    client, embeddings = initialize_chroma()
    
    # Delete existing collection if it exists to start fresh
    try:
        client.delete_collection(name="documents")
        print("Cleared existing collection")
    except:
        pass  # Collection doesn't exist yet, which is fine
    
    # Create fresh collection
    collection = client.create_collection(name="documents")
    print("Created new Chroma collection")
    
    # Load and split documents
    text = load_pdfs_from_directory(pdf_dir)
    
    if not text:
        print("❌ No PDFs found in directory")
        print("=== End Document Loading ===\n")
        return collection
    
    print(f"✅ PDF loaded successfully ({len(text)} characters)")
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    
    print(f"✅ Split into {len(chunks)} document chunks")
    
    # Embed and store in Chroma
    for i, chunk in enumerate(chunks):
        # Get embedding from OpenAI
        embedding = embeddings.embed_query(chunk)
        
        # Add to Chroma
        collection.add(
            ids=[f"doc_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": "resume", "chunk_index": i}]
        )
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks...")
    
    final_count = collection.count()
    print(f"✅ Stored {final_count} chunks in Chroma")
    print("=== End Document Loading ===\n")
    return collection


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant context from Chroma for a query."""
    print(f"\n=== RAG Retrieval Debug ===")
    print(f"Query: {query}")
    
    client, embeddings = initialize_chroma()
    
    try:
        collection = client.get_collection(name="documents")
    except:
        print("No documents in Chroma yet")
        return ""
    
    doc_count = collection.count()
    print(f"Documents in Chroma: {doc_count}")
    
    if doc_count == 0:
        print("Collection is empty!")
        return ""
    
    # Get embedding for query
    query_embedding = embeddings.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Search Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print(f"Results returned: {len(results['documents'][0]) if results['documents'] else 0}")
    
    # Combine results into context string
    context = ""
    if results and results["documents"]:
        for i, doc_list in enumerate(results["documents"]):
            for j, doc in enumerate(doc_list):
                print(f"\n--- Retrieved Document {j+1} ---")
                print(f"Content (first 200 chars): {doc[:200]}...")
                print(f"Metadata: {results['metadatas'][i][j] if results.get('metadatas') else 'N/A'}")
                context += doc + "\n\n"
    
    print(f"\nTotal context length: {len(context)} characters")
    print("=== End RAG Retrieval ===\n")

    return context
