from sentence_transformers import SentenceTransformer
from lib.embed.embedders.baseEmbedder import BaseEmbedder

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