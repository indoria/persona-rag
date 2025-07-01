from typing import List, Dict, Any
import os
from lib.embed.embedders.baseEmbedder import BaseEmbedder
from lib.embed.embedders.azureOpenAIEmbedder import AzureOpenAIEmbedder

def search_similar_chunks(query: str, embedder: BaseEmbedder, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        embedding = embedder._encode([query])[0]

        results = embedder.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })

        return matches

    except Exception as e:
        print(f"❌ Error during similarity search: {e}")
        return []


if __name__ == "__main__":
    deployment_name = os.getenv('AZURE_AI_DEPLOYMENT_NAME', 'text-embedding-ada-002')
    embedder = AzureOpenAIEmbedder(deployment_name=deployment_name)
    matches = search_similar_chunks("Apple released a new IPhone. It is the best IPhone to be released.", embedder)
    print(matches)