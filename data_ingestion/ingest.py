import os
import logging
import json
from patch.sqlite3 import sqlite3

logging.basicConfig(
    filename='ingest.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = "persona.db"

def init_db(db_path=DB_PATH, schema_path="db/schema.sql"):
    """Create DB and tables if not exist, using schema.sql."""
    conn = sqlite3.connect(db_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()

def get_or_create_journalist(conn, name, bio="", pic="", role=""):
    c = conn.cursor()
    c.execute("SELECT id FROM journalists WHERE name=?", (name,))
    row = c.fetchone()
    if row:
        return row[0]
    c.execute("INSERT INTO journalists (name, bio, pic, role) VALUES (?, ?, ?, ?)", (name, bio, pic, role))
    conn.commit()
    return c.lastrowid

def ingest_persona_data(db_journalist_id, json_data, db_file):
    """
    Ingests persona data from a JSON object into the SQLite database.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        print(f"Inserted journalist '{json_data['name']}' with DB ID: {db_journalist_id}")

        # tone
        tone_data = json_data.get('tone', {})
        if tone_data:
            try:
                cursor.execute(
                    "INSERT INTO journalist_tones (journalist_id, voice, style, formality) VALUES (?, ?, ?, ?)",
                    (db_journalist_id, tone_data.get('voice'), tone_data.get('style'), tone_data.get('formality'))
                )
                logger.info(f"Ingested {'tone'} for journalist {json_data['name']}")
            except sqlite3.Error:
                logger.info(f"Failed : Ingesting {'tone'} for journalist {json_data['name']}")

        # interests
        interests = json_data.get('interests', [])
        for interest in interests:
            try:
                cursor.execute(
                    "INSERT INTO journalist_interests (journalist_id, interest_topic) VALUES (?, ?)",
                    (db_journalist_id, interest)
                )
                logger.info(f"Ingested {'interests'} for journalist {json_data['name']}")
            except sqlite3.IntegrityError:
                print(f"  Skipping duplicate interest for {json_data['name']}: {interest}")
                logger.info(f"Failed : Ingesting {'interests'} for journalist {json_data['name']}")


        # aversions
        aversions = json_data.get('aversions', [])
        for aversion in aversions:
            try:
                cursor.execute(
                    "INSERT INTO journalist_aversions (journalist_id, aversion_topic) VALUES (?, ?)",
                    (db_journalist_id, aversion)
                )
                logger.info(f"Ingested {'aversions'} for journalist {json_data['name']}")
            except sqlite3.IntegrityError:
                print(f"  Skipping duplicate aversion for {json_data['name']}: {aversion}")
                logger.info(f"Failed : Ingesting {'aversions'} for journalist {json_data['name']}")

        # evaluation criteria
        eval_criteria = json_data.get('evaluation_criteria', {})
        for criterion_name, details in eval_criteria.items():
            try:
                cursor.execute(
                    "INSERT INTO journalist_evaluation_criteria (journalist_id, criterion_name, question, weight) VALUES (?, ?, ?, ?)",
                    (db_journalist_id, criterion_name, details.get('question'), details.get('weight'))
                )
                logger.info(f"Ingested {'evaluation criteria'} for journalist {json_data['name']}")
            except sqlite3.IntegrityError:
                print(f"  Skipping duplicate evaluation criterion for {json_data['name']}: {criterion_name}")
                logger.info(f"Failed : Ingesting {'evaluation criteria'} for journalist {json_data['name']}")


        # report format preferences
        report_formats = json_data.get('report_format_preferences', [])
        for format_name in report_formats:
            try:
                cursor.execute(
                    "INSERT INTO journalist_report_format_preferences (journalist_id, format_name) VALUES (?, ?)",
                    (db_journalist_id, format_name)
                )
                logger.info(f"Ingested {'report format preferences'} for journalist {json_data['name']}")
            except sqlite3.IntegrityError:
                print(f"  Skipping duplicate report format for {json_data['name']}: {format_name}")
                logger.info(f"Failed : Ingesting {'report format preferences'} for journalist {json_data['name']}")

        # influences
        influences = json_data.get('influences', [])
        for influence in influences:
            try:
                cursor.execute(
                    "INSERT INTO journalist_influences (journalist_id, influence_description) VALUES (?, ?)",
                    (db_journalist_id, influence)
                )
                logger.info(f"Ingested {'influences'} for journalist {json_data['name']}")
            except sqlite3.IntegrityError:
                print(f"  Skipping duplicate influence for {json_data['name']}: {influence}")
                logger.info(f"Failed : Ingesting {'influences'} for journalist {json_data['name']}")

        # current focus
        current_focus = json_data.get('current_focus', [])
        for focus_topic in current_focus:
            try:
                cursor.execute(
                    "INSERT INTO journalist_current_focus (journalist_id, focus_topic) VALUES (?, ?)",
                    (db_journalist_id, focus_topic)
                )
                logger.info(f"Ingested {'current focus'} for journalist {json_data['name']}")
            except sqlite3.IntegrityError:
                print(f"  Skipping duplicate current focus for {json_data['name']}: {focus_topic}")
                logger.info(f"Failed : Ingesting {'current focus'} for journalist {json_data['name']}")

        # response formats
        response_format_data = json_data.get('response_format', {})
        response_format_id = None
        if response_format_data and response_format_data.get('structure'):
            cursor.execute(
                "INSERT INTO journalist_response_formats (journalist_id, structure) VALUES (?, ?)",
                (db_journalist_id, response_format_data.get('structure'))
            )
            response_format_id = cursor.lastrowid # Get the ID for the response format entry

            # response format sections
            sections = response_format_data.get('sections', [])
            for i, section_name in enumerate(sections):
                try:
                    cursor.execute(
                        "INSERT INTO journalist_response_format_sections (response_format_id, section_name, section_order) VALUES (?, ?, ?)",
                        (response_format_id, section_name, i)
                    )
                except sqlite3.IntegrityError:
                    print(f"  Skipping duplicate response format section for {json_data['name']}: {section_name}")

        conn.commit()
        print(f"Successfully ingested data for '{json_data['name']}'.")

    except sqlite3.IntegrityError as e:
        print(f"Data integrity error during ingestion: {e}. This might mean a UNIQUE constraint violation (e.g., journalist name already exists).")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred during ingestion: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def ingest_corpus(corpus_dir="data_ingestion/corpus"):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    persona_dir = 'data_ingestion/persona'

    for persona_name in os.listdir(corpus_dir):
        persona_path = os.path.join(corpus_dir, persona_name)
        if not os.path.isdir(persona_path):
            continue  # skip files in the root corpus_dir

        persona_file = os.path.join(persona_dir, persona_name + ".json")
        with open(persona_file, "r", encoding="utf-8") as f:
            persona_data = json.load(f)
            journalist_id = get_or_create_journalist(conn, persona_data['name'], persona_data['bio'], persona_data['pic'], persona_data['role'])
            ingest_persona_data(journalist_id, persona_data, "persona.db")
            
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
                    logger.info(f"Ingested document {fname} for journalist {persona_data['name']}")
    conn.close()
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    ingest_corpus()