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
# NOTE: Installs the latest cu128 nightly build dynamically from PyTorch's
# index without pinning an ephemeral daily datestamp that gets pruned.
RUN pip install --no-cache-dir --pre \
    torch \
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

EXPOSE 8000
EXPOSE 8501

# No curl in this slim image -- use Python's stdlib for the healthcheck
# instead of adding a package just for this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=3)" || exit 1

CMD ["python", "-m", "app.api.main"]
