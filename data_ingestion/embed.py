from patch.sqlite3 import sqlite3
import chromadb
from uuid import uuid4
import os
from dotenv import load_dotenv
from typing import List
from lib.chunk.tiktoken_chunker import chunk_text
from lib.embed.embedders.azureOpenAIEmbedder import AzureOpenAIEmbedder

load_dotenv()

DB_PATH = os.getenv('DB_PATH')
CHROMA_PATH = os.getenv('CHROMA_PATH')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

client = chromadb.PersistentClient(CHROMA_PATH)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = client.create_collection(COLLECTION_NAME)

def embed_all_corpus_documents():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, journalist_id, content FROM corpus_documents")
    rows = c.fetchall()
    for doc_id, journalist_id, content in rows:
        store_embedding_to_chroma(content, doc_id, journalist_id)
    conn.close()
    print(f"Embedded {len(rows)} documents into ChromaDB.")


def azureEmbedder():
    deployment_name = os.getenv('AZURE_AI_DEPLOYMENT_NAME', 'text-embedding-ada-002')

    embedder = AzureOpenAIEmbedder(deployment_name=deployment_name)
    texts = ["hello world"]
    embedding: List[List[float]] = embedder._encode(texts)

    print("Embedding for 'hello world':")
    print(embedding[0])

def store_embedding_to_chroma(text: str, doc_id: str, journalist_id: str):
    deployment_name = os.getenv('AZURE_AI_DEPLOYMENT_NAME', 'text-embedding-ada-002')

    chunks = chunk_text(text)
    embedder = AzureOpenAIEmbedder(deployment_name=deployment_name)
    for chunk in chunks:
        embedding: List[List[float]] = embedder._encode([chunk])
        collection.add(
            documents=[chunk],
            embeddings=[embedding[0]],
            ids=[str(uuid4())]  # Unique ID per chunk
        )

if __name__ == "__main__":
    #azureEmbedder()
    embed_all_corpus_documents()