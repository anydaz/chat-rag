import os
from dotenv import load_dotenv
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Load environment variables BEFORE any other imports
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.chat import router as chat_router
from .routes.upload import router as upload_router

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

print(f"OpenAI API Key loaded successfully")

# Initialize FastAPI app
app = FastAPI(title="RAG Chat API", version="0.1.0", root_path="/api")

# Add middleware for trusted hosts (optional, can be configured as needed)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Configure CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router)
app.include_router(chat_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

