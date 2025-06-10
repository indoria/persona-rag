import os
import sqlite3
import logging

logging.basicConfig(
    filename='ingest.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = "persona.db"

def init_db(db_path=DB_PATH):
    """Create DB and tables if not exist."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create journalists table
    c.execute("""
        CREATE TABLE IF NOT EXISTS journalists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            bio TEXT
        );
    """)
    # Create corpus_documents table
    c.execute("""
        CREATE TABLE IF NOT EXISTS corpus_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            journalist_id INTEGER,
            content TEXT,
            FOREIGN KEY(journalist_id) REFERENCES journalists(id)
        );
    """)
    conn.commit()
    conn.close()

def get_or_create_journalist(conn, name, bio=""):
    c = conn.cursor()
    c.execute("SELECT id FROM journalists WHERE name=?", (name,))
    row = c.fetchone()
    if row:
        return row[0]
    c.execute("INSERT INTO journalists (name, bio) VALUES (?, ?)", (name, bio))
    conn.commit()
    return c.lastrowid

def ingest_corpus_flat_dir(corpus_dir="data_ingestion/corpus"):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    for fname in os.listdir(corpus_dir):
        if fname.endswith(".txt"):
            # Infer journalist name from filename, e.g. "alice_article1.txt" → "Alice"
            journalist_name = fname.split("_")[0].capitalize()
            bio = f"Journalist persona for {journalist_name}"
            journalist_id = get_or_create_journalist(conn, journalist_name, bio)
            with open(os.path.join(corpus_dir, fname), encoding="utf-8") as f:
                content = f.read().strip()
            c = conn.cursor()
            c.execute(
                "INSERT INTO corpus_documents (journalist_id, content) VALUES (?, ?)",
                (journalist_id, content)
            )
            conn.commit()
            logger.info(f"Ingested document {fname} for journalist {journalist_name}")
    conn.close()
    logger.info("Ingestion complete.")

def ingest_corpus(corpus_dir="data_ingestion/corpus"):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    for persona_name in os.listdir(corpus_dir):
        persona_path = os.path.join(corpus_dir, persona_name)
        if not os.path.isdir(persona_path):
            continue  # skip files in the root corpus_dir
        journalist_name = persona_name.capitalize()
        bio = f"Journalist persona for {journalist_name}"
        journalist_id = get_or_create_journalist(conn, journalist_name, bio)
        for fname in os.listdir(persona_path):
            if fname.endswith(".txt"):
                file_path = os.path.join(persona_path, fname)
                with open(file_path, encoding="utf-8") as f:
                    content = f.read().strip()
                c = conn.cursor()
                c.execute(
                    "INSERT INTO corpus_documents (journalist_id, content) VALUES (?, ?)",
                    (journalist_id, content)
                )
                conn.commit()
                logger.info(f"Ingested document {fname} for journalist {journalist_name}")
    conn.close()
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    ingest_corpus()