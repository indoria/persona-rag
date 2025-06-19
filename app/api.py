import logging
from flask import Flask, request, jsonify

from patch.sqlite3 import sqlite3

import chromadb

from app.pitch_analysis import analyze_pitch
from app.persona_engine import generate_persona_response

DB_PATH = "persona.db"
CHROMA_PATH = "pr_journalist_chroma"

def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    return conn

def get_chroma_client():
    return chromadb.PersistentClient(CHROMA_PATH)

app = Flask(__name__)

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

@app.route("/analyze_pitch", methods=["POST"])
def analyze_pitch_api():
    data = request.json
    pitch_text = data.get("pitch_text", "")
    analysis = analyze_pitch(pitch_text)
    return jsonify(analysis)

@app.route("/journalists", methods=["GET"])
def journalists():
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT id, name, bio, pic, role FROM journalists")
    rows = c.fetchall()
    conn.close()
    res = [{"id": row[0], "name": row[1], "bio": row[2], "pic": row[3], "role": row[4]} for row in rows]
    logger.info(f"Fetched journalists: {len(res)} found")
    return jsonify(res)

def _generate_response(journalist_id, pitch_text):
    db_conn = None
    try:
        db_conn = get_db_conn()
        chroma_client = get_chroma_client()
        response = generate_persona_response(
            journalist_id=journalist_id,
            pitch_text=pitch_text,
            db_conn=db_conn,
            chroma_client=chroma_client,
            num_context=1,
            max_length=128,
            temperature=0.8,
        )
        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Error generating persona response for journalist_id={journalist_id}: {str(e)}")
        return {"status": "error", "message": f"Failed to generate response: {str(e)}"}
    finally:
        if db_conn:
            db_conn.close()

@app.route("/generate_response", methods=["POST"])
def generate_response():
    data = request.json
    journalist_ids = data.get("journalist_ids")
    pitch_text = data.get("pitch_text")
    
    logger.info(f"Received generate_responses for journalist_ids={journalist_ids}")

    if not isinstance(journalist_ids, list) or not journalist_ids:
        logger.warning("Missing or invalid 'journalist_ids' (expected a non-empty list) in request")
        return jsonify({"error": "An array of 'journalist_ids' is required"}), 400
    if not pitch_text:
        logger.warning("Missing 'pitch_text' in request")
        return jsonify({"error": "'pitch_text' is required"}), 400

    results = {}
    for j_id in journalist_ids:
        current_journalist_id = str(j_id)
        logger.info(f"Processing response for individual journalist_id: {current_journalist_id}")
        results[current_journalist_id] = _generate_response(current_journalist_id, pitch_text)

    logger.info("Finished processing all journalist responses.")
    return jsonify(results), 200
