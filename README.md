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

### POST /chat

Send a message to the chat endpoint and receive a response from OpenAI.

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
