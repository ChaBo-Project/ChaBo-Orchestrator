import os
import sys
import json
import time
import argparse
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector, Modifier,
)

# path fix for vector names import (ad hoc script)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from components.retriever.retriever_orchestrator import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

def run_upload():
    # --- 1. Argument Parsing for Generality ---
    parser = argparse.ArgumentParser(description="Upload Parquet data to Qdrant")
    parser.add_argument("--file", type=str, default="data/data.parquet",
                        help="Path to the parquet file (relative to root)")
    parser.add_argument("--collection", type=str, default="my_collection",
                        help="Name of the Qdrant collection")
    parser.add_argument("--vector_size", type=int, default=1024,
                        help="Size of the embedding vectors (e.g., 1024 for BGE-large)")

    # --- Hybrid retrieval options ---
    parser.add_argument("--hybrid", action="store_true",
                        help="Build a hybrid collection: a named dense vector plus a named BM25 "
                             "sparse vector, for [retrieval] hybrid_enabled = true")
    parser.add_argument("--sparse_model", type=str, default="Qdrant/bm25",
                        help="fastembed sparse model id. Must match [retrieval] sparse_model")
    parser.add_argument("--sparse_language", type=str, default="english",
                        help="Snowball stemmer language. Must match [retrieval] sparse_language")
    parser.add_argument("--sparse_batch_size", type=int, default=256,
                        help="Batch size for BM25 encoding")

    args = parser.parse_args()

    # --- 2. Dynamic Connection Logic ---
    # When running manually on host, use localhost. Inside Docker, use qdrant.
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port, prefer_grpc=False)

    # --- 3. Load Data ---
    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        return

    print(f"Reading {args.file}...")
    df = pd.read_parquet(args.file)

    # --- 4. Collection Setup ---
    if not client.collection_exists(args.collection):
        print(f"Creating collection: {args.collection}...")
        if args.hybrid:
            client.create_collection(
                collection_name=args.collection,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(size=args.vector_size, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    # Qdrant computes IDF across the collection at query time; the query side
                    # only sends term presence.
                    SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
                },
            )
            print(f"  hybrid: dense='{DENSE_VECTOR_NAME}' (size={args.vector_size}) "
                  f"+ sparse='{SPARSE_VECTOR_NAME}' (BM25, modifier=IDF)")
        else:
            client.recreate_collection(
                collection_name=args.collection,
                vectors_config=VectorParams(size=args.vector_size, distance=Distance.COSINE),
            )
    else:
        print(f"Collection {args.collection} already exists. Upserting new data...")
        if args.hybrid:
            # A collection created without --hybrid holds one UNNAMED dense vector and no
            # sparse vector. Upserting hybrid points into it fails server-side with an
            # opaque vector-name error, so check the shape and say something useful instead.
            params = client.get_collection(args.collection).config.params
            vectors = params.vectors
            sparse = params.sparse_vectors or {}
            is_hybrid_shaped = (
                isinstance(vectors, dict)
                and DENSE_VECTOR_NAME in vectors
                and SPARSE_VECTOR_NAME in sparse
            )
            if not is_hybrid_shaped:
                print(f"Collection '{args.collection}' exists but is not hybrid-shaped "
                      f"(expected dense='{DENSE_VECTOR_NAME}', sparse='{SPARSE_VECTOR_NAME}').")
                print("Hybrid needs a fresh collection. Either delete this one, or pass a new")
                print("--collection name and update [qdrant] collection in params.cfg.")
                return

    # --- 5. Data Transformation ---
    # Collect rows that parse cleanly, then encode
    print("Preparing points...")
    rows = []  # (point_id, text, metadata, dense_vector)
    for i in range(len(df)):
        try:
            payload_data = df.payload.iloc[i]

            # Robust metadata handling
            metadata = payload_data.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            rows.append((i, payload_data.get('text', ""), metadata, df.vector.iloc[i].tolist()))
        except Exception as e:
            print(f"Skipping row {i} due to error: {e}")

    sparse_vectors = None
    if args.hybrid:
        from fastembed import SparseTextEmbedding

        print("=" * 78)
        print(f"BM25 encoding with model='{args.sparse_model}' language='{args.sparse_language}'")
        print("These MUST match [retrieval] sparse_model / sparse_language in params.cfg.")
        print("=" * 78)

        encoder = SparseTextEmbedding(model_name=args.sparse_model, language=args.sparse_language)
        sparse_vectors = []
        for n, emb in enumerate(encoder.embed([r[1] for r in rows], batch_size=args.sparse_batch_size)):
            sparse_vectors.append(SparseVector(
                indices=[int(x) for x in emb.indices],
                values=[float(v) for v in emb.values],
            ))
            if n and n % 1000 == 0:
                print(f"  ...encoded {n} / {len(rows)}")

        if len(sparse_vectors) != len(rows):
            print(f"Sparse encoder returned {len(sparse_vectors)} vectors for {len(rows)} rows. Aborting.")
            return

        empty = sum(1 for sv in sparse_vectors if not sv.indices)
        if empty:
            print(f"{empty} document(s) encoded to an empty sparse vector (no indexable terms "
                  f"after stopword removal). They can never match a sparse query.")

    points = []
    for n, (point_id, text, metadata, dense_vector) in enumerate(rows):
        vector = (
            {DENSE_VECTOR_NAME: dense_vector, SPARSE_VECTOR_NAME: sparse_vectors[n]}
            if args.hybrid else dense_vector
        )
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": text,
                    "metadata": metadata
                }
            )
        )

    # --- 6. Batch Upload ---
    BATCH_SIZE = 100
    print(f"Starting upload of {len(points)} points to '{args.collection}'...")
    
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(
            collection_name=args.collection,
            points=batch
        )
        if i % 500 == 0:
            print(f"  ...pushed {i} points")
        time.sleep(0.05)

    print(f"Successfully pushed {len(points)} points to Qdrant collection '{args.collection}'.")

if __name__ == "__main__":
    run_upload()