#CHATFED_ORCHESTRATOR

FROM python:3.11.11

# ---------- Create Non-Root User ----------
# Ensures proper file permissions for dev and runtime
RUN useradd -m -u 1000 user

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the BM25 assets used by hybrid retrieval (Qdrant/bm25: stopword lists for its 18
# languages, ~128 KB), so hybrid startup needs no HF Hub access and doesn't re-download
# on every restart. Other sparse models still download at runtime into this cache.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')" && \
    chmod -R a+rwX /opt/fastembed_cache

# ---------- Copy Project Files ----------
# Set appropriate ownership and permissions
COPY --link --chown=1000 . .

# fastapi and gradio
EXPOSE 7860 

# launch with unbuffered output
# Use main_prod.py for production with full RAG functionality
# Use main.py for testing mode with startup validation
CMD ["python", "-u", "src/main.py"]


