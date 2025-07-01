import chromadb
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    def __init__(self, chroma_path: str, collection_name: str = "corpus_embeddings"):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Connected to existing ChromaDB collection: '{self.collection_name}' at {self.chroma_path}")
        except Exception:
            print(f"ChromaDB collection '{self.collection_name}' not found, creating it at {self.chroma_path}.")
            self.collection = self.client.create_collection(self.collection_name)

    @abstractmethod
    def _encode(self, texts: list[str]) -> list[list[float]]:
        pass

    def store_embedding(self, doc_id: str, journalist_id: str, text: str, metadata: dict = {}):
        unique_id = f"{journalist_id}_{doc_id}"
        existing_ids = self.collection.get(ids=[unique_id], include=[])['ids']
        if existing_ids:
            print(f"Document with ID '{unique_id}' already exists in ChromaDB. Skipping.")
            return
        try:
            embedding = self._encode([text])[0]
            metadata = { "doc_id": doc_id, "journalist_id": journalist_id, **metadata }
            self.collection.add(
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[unique_id]
            )
            print(f"Added embedding for doc_id '{doc_id}' by journalist '{journalist_id}'.")
        except Exception as e:
            print(f"Error storing embedding for doc_id '{doc_id}': {e}")

    def query_embeddings(self, query_text: str, n_results: int = 5):
        try:
            query_embedding = self._encode([query_text])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'distances', 'metadatas']
            )
            return results
        except Exception as e:
            print(f"Error querying embeddings: {e}")
            return None