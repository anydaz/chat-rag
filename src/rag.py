"""RAG module for document retrieval with Chroma."""

import os
import chromadb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from .documents import DocumentLoader




def initialize_chroma():
    """Initialize Chroma vector store with OpenAI embeddings."""
    # Create persistent Chroma client
    client = chromadb.PersistentClient(path="./data/chroma")
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    return client, embeddings


def load_documents_to_chroma(pdf_dir: str = "data/pdfs", docs_dir: str = "data/docs"):
    """Load documents from multiple directories and store embeddings in Chroma."""
    print(f"\n=== Loading Documents to Chroma ===")
    print(f"PDF Directory: {pdf_dir}")
    print(f"Docs Directory: {docs_dir}")
    
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
    
    # Load documents from both directories with metadata
    all_documents = []
    
    # Load from PDFs directory
    print(f"\n📄 Loading from {pdf_dir}...")
    pdf_loader = DocumentLoader(pdf_dir)
    pdf_docs = pdf_loader.load_all_documents()
    all_documents.extend(pdf_docs)
    
    # Load from docs directory
    print(f"\n📚 Loading from {docs_dir}...")
    docs_loader = DocumentLoader(docs_dir)
    docs_docs = docs_loader.load_all_documents()
    all_documents.extend(docs_docs)
    
    if not all_documents:
        print("❌ No documents found in any directory")
        print("=== End Document Loading ===\n")
        return collection
    
    # Combine text from all documents
    combined_text = ""
    doc_metadata_map = {}  # Map to track which chunks came from which document
    
    for doc_text, metadata in all_documents:
        combined_text += doc_text + "\n"
        doc_metadata_map[len(combined_text)] = metadata  # Store position and metadata
    
    print(f"\n✅ All documents loaded successfully ({len(combined_text)} characters)")
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(combined_text)
    
    print(f"✅ Split into {len(chunks)} document chunks")
    
    # Embed and store in Chroma with source metadata
    for i, chunk in enumerate(chunks):
        # Get embedding from OpenAI
        embedding = embeddings.embed_query(chunk)
        
        # Find which source document this chunk came from
        source_metadata = {"document_type": "unknown"}
        for pos, metadata in doc_metadata_map.items():
            if pos >= len(combined_text[:combined_text.find(chunk) + len(chunk)]):
                source_metadata = metadata
                break
        
        # Add to Chroma with comprehensive metadata
        collection.add(
            ids=[f"doc_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "chunk_index": i,
                "filename": source_metadata.get("filename", "unknown"),
                "file_type": source_metadata.get("file_type", "unknown"),
                "document_type": source_metadata.get("document_type", "unknown"),
                "directory": source_metadata.get("directory", "unknown"),
                "source_path": source_metadata.get("source_path", "unknown"),
            }]
        )
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks...")
    
    final_count = collection.count()
    print(f"✅ Stored {final_count} chunks in Chroma with source metadata")
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
                metadata = results['metadatas'][i][j] if results.get('metadatas') else {}
                print(f"\n--- Retrieved Document {j+1} ---")
                print(f"Filename: {metadata.get('filename', 'unknown')}")
                print(f"Type: {metadata.get('document_type', 'unknown')}")
                print(f"Content (first 200 chars): {doc[:200]}...")
                context += doc + "\n\n"
    
    print(f"\nTotal context length: {len(context)} characters")
    print("=== End RAG Retrieval ===\n")

    return context
