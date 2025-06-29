from patch.sqlite3 import sqlite3
import os
import chromadb
from dotenv import load_dotenv
from db.chroma_embedders import SentenceTransformerEmbedder, OpenAIEmbedder
from lib.chunk.tiktoken_chunker import chunk_text
from lib.embed.embedders.OpenAI import get_embedding
from uuid import uuid4

load_dotenv()

DB_PATH = os.getenv('DB_PATH')
CHROMA_PATH = os.getenv('CHROMA_PATH')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

client = chromadb.PersistentClient(CHROMA_PATH)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = client.create_collection(COLLECTION_NAME)

def add_to_chroma():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, journalist_id, content FROM corpus_documents")
    rows = c.fetchall()

    for doc_id, journalist_id, content in rows:
        chunks = chunk_text(content)
        metadata = {"doc_id": doc_id, "journalist_id": journalist_id}
        for chunk in chunks:
            embedding = get_embedding(chunk)
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[str(uuid4())]  # Unique ID per chunk
                #ids=[f"{journalist_id}_{doc_id}"]
            )

    conn.close()
    print(f"Embedded {len(rows)} documents into ChromaDB.")


if __name__ == "__main__":
    add_to_chroma()