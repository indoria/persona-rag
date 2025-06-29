from typing import List
from openai import OpenAI
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from lib.embed.embedders.baseEmbedder import BaseEmbedder
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv('DB_PATH')
CHROMA_PATH = os.getenv('CHROMA_PATH')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
AZURE_AI_FOUNDRY_KEY = os.getenv('AZURE_AI_FOUNDRY_KEY')
AZURE_AI_FOUNDRY_ENDPOINT = os.getenv('AZURE_AI_FOUNDRY_ENDPOINT')


class AzureOpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        azure_api_key: str = AZURE_AI_FOUNDRY_KEY,
        azure_endpoint: str = AZURE_AI_FOUNDRY_ENDPOINT,
        deployment_name: str,
        api_version: str = "2023-12-01-preview",
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME
    ):
        super().__init__(chroma_path, collection_name)
        self.deployment_name = deployment_name
        self.client_aoai = OpenAI(
            api_key=azure_api_key,
            base_url=f"{azure_endpoint}/openai/deployments/{deployment_name}",
            default_headers={
                "api-key": azure_api_key
            },
            api_version=api_version,
            api_type="azure"
        )

    def _encode(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client_aoai.embeddings.create(
                input=texts,
                engine=self.deployment_name
            )
            return [record.embedding for record in response.data]
        except Exception as e:
            print(f"Error generating embeddings with Azure OpenAI: {e}")
            raise
