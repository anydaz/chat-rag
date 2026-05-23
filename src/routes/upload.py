"""File upload routes for documents and PDFs."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import shutil

# Initialize router
router = APIRouter()

# Define data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDFS_DIR = DATA_DIR / "pdfs"
DOCS_DIR = DATA_DIR / "docs"

# Create directories if they don't exist
PDFS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file to data/pdfs directory."""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save file
        file_path = PDFS_DIR / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "message": f"PDF '{file.filename}' uploaded successfully",
            "filename": file.filename,
            "location": str(file_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.post("/upload/doc")
async def upload_doc(file: UploadFile = File(...)):
    """Upload a document file (txt, docx, md) to data/docs directory."""
    try:
        # Validate file type
        allowed_extensions = {'.txt', '.docx', '.md', '.doc'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Only {', '.join(allowed_extensions)} files are allowed"
            )
        
        # Save file
        file_path = DOCS_DIR / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' uploaded successfully",
            "filename": file.filename,
            "location": str(file_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.get("/upload/status")
async def upload_status():
    """Get information about uploaded files."""
    try:
        pdf_files = list(PDFS_DIR.glob("*.pdf")) if PDFS_DIR.exists() else []
        doc_files = list(DOCS_DIR.glob("*")) if DOCS_DIR.exists() else []
        doc_files = [f for f in doc_files if f.is_file()]
        
        return {
            "pdfs": {
                "count": len(pdf_files),
                "files": [f.name for f in pdf_files]
            },
            "docs": {
                "count": len(doc_files),
                "files": [f.name for f in doc_files]
            },
            "total": len(pdf_files) + len(doc_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching status: {str(e)}")
