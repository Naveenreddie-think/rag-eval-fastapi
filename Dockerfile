# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# git: needed if scripts/ingest.py's sparse-checkout is ever run inside the
# container. build-essential: covers any source builds pulled in transitively
# by the pinned wheels below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch: NIGHTLY cu128 build, installed explicitly and FIRST ---
#
# This project targets an RTX 5060 (Blackwell / sm_120). Stable PyTorch
# releases do not support sm_120 -- only nightly cu128 builds do (see
# app/generation/local_llm.py's docstring and FINDINGS.md for the original
# diagnosis). If torch were left to requirements.txt / the default PyPI
# index, pip would silently resolve a stable release that either falls back
# to CPU or errors at kernel-launch time instead of failing loudly at
# install time -- so it's pinned to the exact nightly build verified
# working on this project, installed from PyTorch's own nightly index.
#
# NOTE: PyTorch prunes older nightly wheels from this index over time.
# Checked directly against the index before pinning (not guessed): as of
# this build, 2.12.0.dev20260408+cu128 is the ONLY cu128 nightly currently
# listed for cp312/manylinux -- there is no newer one to pin instead. If
# this exact version 404s on a future build, check
# https://download.pytorch.org/whl/nightly/cu128/torch/ for what's
# actually available, update the pin, and re-verify
# `torch.cuda.is_available()` before trusting it -- do not silently fall
# back to a stable release or leave this unpinned.
RUN pip install --no-cache-dir --pre \
    torch==2.12.0.dev20260408+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Everything else from the normal index. torch is already satisfied above
# (exact version match), so this will not touch it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the runtime data it needs (chunk index for BM25, cached eval
# results served by /api/metrics). data/raw/ is intentionally NOT copied --
# it's only needed for offline ingestion (scripts/ingest.py), not serving.
COPY app/ app/
COPY data/processed/ data/processed/
COPY streamlit_app.py .
# Fine-tuned reranker (Step 8) -- app/retrieval/hybrid.py falls back to the
# off-the-shelf model if this directory isn't present, so a build context
# without Git LFS objects pulled still works, just without the domain
# adaptation gain.
COPY models/fastapi-reranker-minilm/ models/fastapi-reranker-minilm/

EXPOSE 8000
EXPOSE 8501

# Entrypoint Design:
# - Default CMD launches Streamlit on port 8501 to support direct single-container
#   deployments (specifically Hugging Face Spaces with Docker SDK, which runs the
#   image's default CMD without a runtime command-override mechanism and routes
#   traffic to README.md's app_port: 8501).
# - Local multi-service development is managed via docker-compose.yml, which explicitly
#   overrides `command:` for both the FastAPI backend (`api` on port 8000) and the
#   Streamlit frontend (`ui` on port 8501).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
