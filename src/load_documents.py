#!/usr/bin/env python3
"""Standalone script to load documents into Chroma vector store."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.rag import load_documents_to_chroma

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    print("🚀 Starting document loading process...\n")
    
    try:
        load_documents_to_chroma()
        print("\n✅ Document loading completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error loading documents: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
