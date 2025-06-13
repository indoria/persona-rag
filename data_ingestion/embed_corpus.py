try:
    import sys
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
    print("Successfully swapped sqlite3 module with pysqlite3 for ChromaDB compatibility.")
except ImportError:
    print("pysqlite3 not found. Falling back to default sqlite3. ChromaDB might still throw errors.")

import sqlite3
from db.chroma_utils import store_embedding

DB_PATH = "persona.db"

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