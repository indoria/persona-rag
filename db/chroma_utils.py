import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "pr_journalist_chroma"
COLLECTION = "corpus_embeddings"

client = chromadb.PersistentClient(CHROMA_PATH)
try:
    col = client.get_collection(COLLECTION)
except Exception:
    col = client.create_collection(COLLECTION)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def store_embedding(doc_id, journalist_id, text):
    embedding = embedder.encode([text])[0]
    metadata = {"doc_id": doc_id, "journalist_id": journalist_id}
    col.add(
        embeddings=[embedding.tolist()],
        metadatas=[metadata],
        ids=[f"{journalist_id}_{doc_id}"]
    )