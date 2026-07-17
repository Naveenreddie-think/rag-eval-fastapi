"""
Step 7: Serving.

Responsibility: FastAPI app exposing the RAG pipeline as an API
(/query endpoint: question in, answer + citations + latency breakdown
out). Backing for the Streamlit frontend, same pattern as Project 1.

Run with (Windows-safe, matches Project 1 convention):
    python -m app.api.main
(not `python app/api/main.py`, to avoid ModuleNotFoundError from
internal `from app.x import y` imports)
"""
