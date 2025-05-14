import os
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configurations
DATA_DIR = Path(__file__).parent.parent / "data/readme_lifesight/docs/METHODOLOGIES"
QDRANT_COLLECTION = "lifesight_marketing_measurements"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Helper: Get all relevant markdown files
def get_markdown_files(data_dir: Path) -> List[Path]:
    md_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".md"):
                md_files.append(Path(root).joinpath(file))
    return md_files

# Helper: Read and chunk markdown files
def chunk_markdown_file(file_path: Path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [{
        "text": chunk,
        "metadata": {
            "source": str(file_path),
            "chunk_id": i
        }
    } for i, chunk in enumerate(chunks)]

# Main ingestion function
def ingest():
    # 1. Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    # 2. Connect to QDrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # 3. Create collection if not exists
    if QDRANT_COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qdrant_models.VectorParams(size=model.get_sentence_embedding_dimension(), distance=qdrant_models.Distance.COSINE)
        )
    # 4. Process files
    md_files = get_markdown_files(DATA_DIR)
    print(f"Found {len(md_files)} markdown files in {DATA_DIR.resolve()}")
    if len(md_files) > 0:
        print("Sample files:", [str(f) for f in md_files[:3]])
    all_chunks = []
    for md_file in md_files:
        all_chunks.extend(chunk_markdown_file(md_file))
    print(f"Total chunks to ingest: {len(all_chunks)}")
    # 5. Embed and upload
    point_id = 0
    for i in range(0, len(all_chunks), 64):
        batch = all_chunks[i:i+64]
        texts = [item["text"] for item in batch]
        embeddings = model.encode(texts, show_progress_bar=False)
        payload = [item["metadata"] | {"text": item["text"]} for item in batch]
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                qdrant_models.PointStruct(
                    id=point_id + j,
                    vector=emb.tolist(),
                    payload=pl
                ) for j, (emb, pl) in enumerate(zip(embeddings, payload))
            ]
        )
        point_id += len(batch)
    print(f"Ingested {len(all_chunks)} chunks into QDrant collection '{QDRANT_COLLECTION}'")

if __name__ == "__main__":
    ingest() 