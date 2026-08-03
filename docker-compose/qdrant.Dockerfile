# 1.17+ is required for weighted RRF (Rrf(weights=[...])), used by hybrid retrieval.
FROM qdrant/qdrant:v1.17.1

# Create the data directory to ensure permissions are correct
RUN mkdir -p /data/qdrant_data /data/qdrant_snapshots

# Set Qdrant storage paths to the persistent volume location
ENV QDRANT__STORAGE__STORAGE_PATH="/data/qdrant_data"
ENV QDRANT__STORAGE__SNAPSHOTS_PATH="/data/qdrant_snapshots"

# Expose HTTP (6333) and gRPC (6334)
EXPOSE 6333
EXPOSE 6334