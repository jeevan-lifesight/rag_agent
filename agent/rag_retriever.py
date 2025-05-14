from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

class RAGRetriever:
    def __init__(
        self,
        collection_name: str = "lifesight_marketing_measurements",
        embedding_model: str = "all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        top_k: int = 5,
    ):
        self.collection_name = collection_name
        self.model = SentenceTransformer(embedding_model)
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.top_k = top_k

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        query_vector = self.model.encode([query_text])[0]
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.top_k,
        )
        return [
            {
                "score": hit.score,
                "text": hit.payload.get("text"),
                "source": hit.payload.get("source"),
                "chunk_id": hit.payload.get("chunk_id"),
            }
            for hit in search_result
        ]

if __name__ == "__main__":
    retriever = RAGRetriever()
    results = retriever.query("How does causal attribution work in marketing measurement?")
    for i, res in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Score: {res['score']:.3f}")
        print(f"Source: {res['source']} (chunk {res['chunk_id']})")
        print(f"Text: {res['text'][:300]}...\n") 