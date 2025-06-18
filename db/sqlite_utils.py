from patch.sqlite3 import sqlite3

DB_PATH = "persona.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    with open("db/schema.sql", "r") as f:
        schema = f.read()
    conn = get_conn()
    conn.executescript(schema)
    conn.commit()
    conn.close()

def insert_journalist(name, bio):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO journalists (name, bio) VALUES (?, ?)", (name, bio))
    conn.commit()
    c.execute("SELECT id FROM journalists WHERE name=?", (name,))
    jid = c.fetchone()[0]
    conn.close()
    return jid

def insert_document(journalist_id, document_type, content, processed_content, avg_sentence_length, sentiment_score):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO corpus_documents (journalist_id, document_type, content, processed_content, avg_sentence_length, sentiment_score) VALUES (?, ?, ?, ?, ?, ?)",
        (journalist_id, document_type, content, processed_content, avg_sentence_length, sentiment_score),
    )
    conn.commit()
    doc_id = c.lastrowid
    conn.close()
    return doc_id

def insert_interest(journalist_id, interest_topic, strength_score):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO journalist_interests (journalist_id, interest_topic, strength_score) VALUES (?, ?, ?)",
        (journalist_id, interest_topic, strength_score),
    )
    conn.commit()
    conn.close()