# RAG Project with LangGraph

A Python project using LangGraph for building RAG (Retrieval-Augmented Generation) applications.

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Install dependencies with uv:

```bash
uv sync
```

2. Update the `.env` file with your OpenAI API key:

```bash
# .env
OPENAI_API_KEY=your_actual_api_key_here
```

3. Run the FastAPI server:

```bash
uv run src/main.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### POST /chat (Streaming)

Send a message to the chat endpoint and receive a **streamed response** from OpenAI. This provides better UX for chatbots as the response appears incrementally.

**Request:**

```json
{
  "message": "Hello, what is 2+2?"
}
```

**Response:** Streamed text chunks (Server-Sent Events format)

Using curl with streaming:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what is 2+2?"}' \
  -N  # Disable buffering to see streaming response
```

Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Hello, what is 2+2?"},
    stream=True,
)

for chunk in response.iter_content(decode_unicode=True):
    if chunk:
        print(chunk, end="", flush=True)
```

### POST /chat/non-stream (Non-Streaming)

Send a message and get the complete response at once (original behavior).

**Request:**

```json
{
  "message": "Hello, what is 2+2?"
}
```

**Response:**

```json
{
  "message": "Hello, what is 2+2?",
  "response": "2 + 2 equals 4."
}
```

Using curl:

```bash
curl -X POST http://localhost:8000/chat/non-stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what is 2+2?"}'
```

### GET /health

Health check endpoint to verify the server is running.

**Response:**

```json
{
  "status": "ok"
}
```

**Interactive API Documentation:**

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
.
├── pyproject.toml      # uv project configuration
├── .env                # Environment variables (create with your API key)
├── .gitignore          # Git ignore rules
├── README.md           # This file
├── data/
│   ├── .gitignore      # Ignore PDF files in git
│   └── pdfs/           # PDF files for RAG (add your PDFs here)
└── src/
    ├── __init__.py
    ├── main.py         # FastAPI application entry point
    ├── graph.py        # LangGraph definition for chat
    └── routes/
        ├── __init__.py
        └── chat.py     # Chat API routes
```

## Development

Run the server with auto-reload:

```bash
uv run uvicorn src.main:app --reload
```

Run tests:

```bash
uv run pytest
```

Format code:

```bash
uv run black src/
```

Lint:

```bash
uv run ruff check src/
```
