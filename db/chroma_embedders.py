import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
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

    def store_embedding(self, doc_id: str, journalist_id: str, text: str):
        unique_id = f"{journalist_id}_{doc_id}"
        existing_ids = self.collection.get(ids=[unique_id], include=[])['ids']
        if existing_ids:
            print(f"Document with ID '{unique_id}' already exists in ChromaDB. Skipping.")
            return
        try:
            embedding = self._encode([text])[0]
            metadata = {"doc_id": doc_id, "journalist_id": journalist_id}
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

class SentenceTransformerEmbedder(BaseEmbedder):
    # model_name = "all-MiniLM-L6-v2", "BAAI/bge-m3", "all-mpnet-base-v2"
    def __init__(self, model_name: str, chroma_path: str, collection_name: str = "corpus_embeddings"):
        super().__init__(chroma_path, collection_name)
        try:
            self.embedder = SentenceTransformer(model_name)
            print(f"Successfully initialized SentenceTransformer model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model '{model_name}': {e}")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts).tolist()

class OpenAIEmbedder(BaseEmbedder):
    # model_name = "text-embedding-ada-002", "text-embedding-3-small"
    def __init__(self, model_name: str, openai_api_key: str, chroma_path: str, collection_name: str = "corpus_embeddings"):
        super().__init__(chroma_path, collection_name)
        self.model_name = model_name
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be provided for OpenAIEmbedder.")
        self.client_openai = OpenAI(api_key=openai_api_key)
        print(f"Successfully initialized OpenAI Embedder with model: {model_name}")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client_openai.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise RuntimeError(f"Error encoding with OpenAI model '{self.model_name}': {e}")
