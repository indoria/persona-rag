import logging
from flask import Flask, request, jsonify
import sqlite3
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


if __name__ == "__main__":
    app.run(debug=True)

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
    c.execute("SELECT id, name, bio FROM journalists")
    rows = c.fetchall()
    conn.close()
    res = [{"id": row[0], "name": row[1], "bio": row[2]} for row in rows]
    logger.info(f"Fetched journalists: {len(res)} found")
    return jsonify(res)

@app.route("/generate_response", methods=["POST"])
def generate_response():
    data = request.json
    journalist_id = data.get("journalist_id")
    pitch_text = data.get("pitch_text")
    logger.info(f"Received generate_response for journalist_id={journalist_id}")
    if not journalist_id or not pitch_text:
        logger.warning("Missing journalist_id or pitch_text in request")
        return jsonify({"error": "journalist_id and pitch_text required"}), 400
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
        db_conn.close()
        logger.info("Generated persona response successfully")
        return jsonify({"response": response})
    except Exception as e:
        logger.exception(f"Error generating persona response: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
