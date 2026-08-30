# MuleShield AI — single container, single process
# SIH26184 | MHA / I4C
#
#   docker build -t muleshield .
#   docker run -p 7860:7860 muleshield
#
# One process serves the API, the WebSocket and the console. That is not a
# simplification for the demo — it is what the architecture requires. Case
# workflow state, notes and the audit trail live in process memory, so two
# replicas would disagree with each other, and a WebSocket broadcast would only
# reach the clients attached to whichever replica sent it. Scale this up and it
# breaks. Run one.
#
# Port 7860 is Hugging Face Spaces' default; override with PORT anywhere else.

# ── Stage 1: build the console ───────────────────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
# No VITE_API_BASE_URL on purpose. Unset, the bundle uses relative URLs and asks
# the origin it was served from — which is this same container. Setting it here
# would hardcode one host into the build.
RUN npm run build


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

# libgomp is XGBoost's OpenMP runtime; it is not in the slim base and the import
# fails without it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch, installed before everything else.
#
# The default wheel carries the entire CUDA toolchain — several gigabytes of GPU
# runtime for a host that has no GPU. This is the single biggest lever on image
# size. Installing it first also means the layer is cached across rebuilds when
# only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch \
 && pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY engine/ ./engine/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY models/ ./models/
COPY embeddings/ ./embeddings/

# Generate the corpus at BUILD time, not on startup.
#
# transactions.csv (~114 MB) and graph_edges.csv (~40 MB) exceed GitHub's file
# limit and are gitignored, so they are absent from a clone. Generating them
# takes minutes: a container that spends those minutes booting looks broken, and
# on a platform with a startup health check it will be killed before it answers.
# Output is deterministic at seed 42, so baking it into the image is safe.
RUN python scripts/generate_data.py

COPY --from=frontend /build/dist ./frontend/dist

# Fail the build rather than ship an image that starts and then cannot work.
RUN python -c "import torch, xgboost, torch_geometric; print('deps ok')" \
 && test -f data/transactions.csv \
 && test -f frontend/dist/index.html \
 && echo "artefacts ok"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",7860)}/health').read()"

# One worker, deliberately. See the note at the top of this file.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
