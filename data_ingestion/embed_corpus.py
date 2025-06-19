from patch.sqlite3 import sqlite3
from db.chroma_utils import store_embedding
import os
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv('DB_PATH')

def embed_all_corpus_documents():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, journalist_id, content FROM corpus_documents")
    rows = c.fetchall()
    for doc_id, journalist_id, content in rows:
        store_embedding(doc_id, journalist_id, content)
    conn.close()
    print(f"Embedded {len(rows)} documents into ChromaDB.")

if __name__ == "__main__":
    embed_all_corpus_documents()