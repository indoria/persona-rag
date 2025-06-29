from openai import OpenAI
from lib.embed.embedders.baseEmbedder import BaseEmbedder
import os

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