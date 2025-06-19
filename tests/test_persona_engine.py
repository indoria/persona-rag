import os
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv('DB_PATH')
CHROMA_PATH = os.getenv('CHROMA_PATH')

def test_generate_persona_response_smoke():
    from app.persona_engine import generate_persona_response
    from patch.sqlite3 import sqlite3, chromadb
    db_conn = sqlite3.connect(DB_PATH)
    chroma_client = chromadb.PersistentClient(CHROMA_PATH)
    # Use a valid journalist_id from your DB (e.g., 1 or 2 for POC)
    resp = generate_persona_response(1, "Tell me about Tesla's new battery.", db_conn, chroma_client)
    assert isinstance(resp, str)
    assert len(resp) > 0
    db_conn.close()

def test_generate_persona_response_invalid_journalist():
    from app.persona_engine import generate_persona_response
    from patch.sqlite3 import sqlite3, chromadb
    db_conn = sqlite3.connect(DB_PATH)
    chroma_client = chromadb.PersistentClient(CHROMA_PATH)
    # Try a non-existent journalist_id
    resp = generate_persona_response(9999, "Test pitch.", db_conn, chroma_client)
    assert "Error" in resp
    db_conn.close()