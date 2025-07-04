"""
Run this script after ingestion to fine-tune a GPT-2 model for each journalist.
"""
import os
from app.persona_engine import fine_tune_journalist_model
from patch.sqlite3 import sqlite3
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv('DB_PATH')

def get_corpus_for_journalist(jid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT content FROM corpus_documents WHERE journalist_id=?", (jid,))
    docs = [row[0] for row in c.fetchall()]
    conn.close()
    return docs

def get_journalists():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name_full FROM journalists")
    js = c.fetchall()
    conn.close()
    return js

if __name__ == "__main__":
    journalists = get_journalists()
    for jid, name in journalists:
        print(f"Fine-tuning model for journalist: {name}")
        corpus = get_corpus_for_journalist(jid)
        fine_tune_journalist_model(name, corpus, model_name="distilgpt2", epochs=2)