import os
import logging
import json
from patch.sqlite3 import sqlite3
import shutil
import uuid

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    filename = os.getenv('INGEST_LOG'),
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv('DB_PATH')
PERSONA_DIR = 'data_ingestion/persona'
CORPUS_DIR = 'data_ingestion/corpus'
SCHEMA_PATH = "db/schema.sql"

def init_db(recreate=False):
    conn = None
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        if recreate:
            logger.info(f"Attempting to delete existing database file at {DB_PATH} for recreation.")
            print(f"Attempting to delete existing database file at {DB_PATH} for recreation.")
            if os.path.exists(DB_PATH):
                try:
                    os.remove(DB_PATH)
                    logger.info(f"Successfully deleted existing database file: {DB_PATH}")
                    print(f"Successfully deleted existing database file: {DB_PATH}")
                except OSError as e:
                    logger.error(f"Error deleting database file '{DB_PATH}' during recreation: {e}")
                    print(f"Error deleting database file during recreation: {e}")
            else:
                logger.info(f"No existing database file found at {DB_PATH} for recreation.")
                print(f"No existing database file found at {DB_PATH} for recreation.")

        conn = sqlite3.connect(DB_PATH)
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        logger.info(f"Database initialized or verified at {DB_PATH}")
        print(f"Database initialized or verified at {DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        print(f"Error initializing database: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_or_create_journalist(conn, name, bio="", pic=""):
    c = conn.cursor()
    c.execute("SELECT id, bio, pic FROM journalists WHERE name_full=?", (name,))
    row = c.fetchone()
    if row:
        journalist_id, current_bio, current_pic = row
        if current_bio != bio or current_pic != pic:
            c.execute("UPDATE journalists SET bio=?, pic=? WHERE id=?", (bio, pic, journalist_id))
            conn.commit()
            logger.info(f"Journalist '{name}' already exists with ID: {journalist_id}. Basic info updated.")
        else:
            logger.info(f"Journalist '{name}' already exists with ID: {journalist_id}. No basic info update needed.")
        return journalist_id
    else:
        journalist_id = str(uuid.uuid4())
        c.execute("INSERT INTO journalists (journalist_id, name_full, bio, pic) VALUES (?, ?, ?, ?)", (journalist_id, name, bio, pic))
        conn.commit()
        new_id = c.lastrowid
        logger.info(f"Created new journalist '{name}' with ID: {new_id}")
        return new_id

def ingest_persona_data_details(db_journalist_id, json_data, cursor, conn):
    print(f"Ingesting persona data for journalist '{json_data['name']}' (DB ID: {db_journalist_id})...")

    ingestion_map = [
        {'table': 'journalist_tones', 'json_key': 'tone',
         'upsert': True,
         'sql_select': "SELECT id FROM journalist_tones WHERE journalist_id=?",
         'sql_update': "UPDATE journalist_tones SET voice=?, style=?, formality=? WHERE journalist_id=?",
         'sql_insert': "INSERT INTO journalist_tones (journalist_id, voice, style, formality) VALUES (?, ?, ?, ?)",
         'data_fn_insert': lambda d: (db_journalist_id, d.get('voice'), d.get('style'), d.get('formality')),
         'data_fn_update': lambda d: (d.get('voice'), d.get('style'), d.get('formality'), db_journalist_id)},

        {'table': 'journalist_interests', 'json_key': 'interests', 'is_list': True,
         'sql': "INSERT INTO journalist_interests (journalist_id, interest_topic) VALUES (?, ?)",
         'data_fn': lambda item: (db_journalist_id, item)},

        {'table': 'journalist_aversions', 'json_key': 'aversions', 'is_list': True,
         'sql': "INSERT INTO journalist_aversions (journalist_id, aversion_topic) VALUES (?, ?)",
         'data_fn': lambda item: (db_journalist_id, item)},

        {'table': 'journalist_triggers', 'json_key': 'triggers', 'is_list': True,
         'sql': "INSERT INTO journalist_triggers (journalist_id, trigger_topic) VALUES (?, ?)",
         'data_fn': lambda item: (db_journalist_id, item)},

        {'table': 'journalist_evaluation_criteria', 'json_key': 'evaluation_criteria', 'is_dict': True,
         'upsert': True,
         'sql_select': "SELECT id FROM journalist_evaluation_criteria WHERE journalist_id=? AND criterion_name=?",
         'sql_update': "UPDATE journalist_evaluation_criteria SET question=?, weight=? WHERE journalist_id=? AND criterion_name=?",
         'sql_insert': "INSERT INTO journalist_evaluation_criteria (journalist_id, criterion_name, question, weight) VALUES (?, ?, ?, ?)",
         'data_fn_insert': lambda name, details: (db_journalist_id, name, details.get('question'), details.get('weight')),
         'data_fn_update': lambda name, details: (details.get('question'), details.get('weight'), db_journalist_id, name)},

        {'table': 'journalist_report_format_preferences', 'json_key': 'report_format_preferences', 'is_list': True,
         'sql': "INSERT INTO journalist_report_format_preferences (journalist_id, format_name) VALUES (?, ?)",
         'data_fn': lambda item: (db_journalist_id, item)},

        {'table': 'journalist_influences', 'json_key': 'influences', 'is_list': True,
         'sql': "INSERT INTO journalist_influences (journalist_id, influence_description) VALUES (?, ?)",
         'data_fn': lambda item: (db_journalist_id, item)},

        {'table': 'journalist_current_focus', 'json_key': 'current_focus', 'is_list': True,
         'sql': "INSERT INTO journalist_current_focus (journalist_id, focus_topic) VALUES (?, ?)",
         'data_fn': lambda item: (db_journalist_id, item)},
    ]

    for mapping in ingestion_map:
        table_name = mapping['table']
        json_key = mapping['json_key']
        data_to_ingest = json_data.get(json_key)

        if data_to_ingest:
            if mapping.get('upsert'):
                if not mapping.get('is_list') and not mapping.get('is_dict'):
                    cursor.execute(mapping['sql_select'], (db_journalist_id,))
                    exists = cursor.fetchone()
                    try:
                        if exists:
                            cursor.execute(mapping['sql_update'], mapping['data_fn_update'](data_to_ingest))
                            logger.info(f"Updated {json_key} for journalist {json_data['name']}")
                        else:
                            cursor.execute(mapping['sql_insert'], mapping['data_fn_insert'](data_to_ingest))
                            logger.info(f"Inserted {json_key} for journalist {json_data['name']}")
                    except sqlite3.Error as e:
                        logger.error(f"Error ingesting {json_key} for {json_data['name']}: {e}")
                elif mapping.get('is_dict'):
                    for name, details in data_to_ingest.items():
                        cursor.execute(mapping['sql_select'], (db_journalist_id, name))
                        exists = cursor.fetchone()
                        try:
                            if exists:
                                cursor.execute(mapping['sql_update'], mapping['data_fn_update'](name, details))
                                logger.info(f"Updated {json_key} (criterion: {name}) for journalist {json_data['name']}")
                            else:
                                cursor.execute(mapping['sql_insert'], mapping['data_fn_insert'](name, details))
                                logger.info(f"Inserted {json_key} (criterion: {name}) for journalist {json_data['name']}")
                        except sqlite3.Error as e:
                            logger.error(f"Error ingesting {json_key} for {json_data['name']} (criterion: {name}): {e}")
            else:
                if mapping.get('is_list'):
                    for item in data_to_ingest:
                        try:
                            cursor.execute(mapping['sql'], mapping['data_fn'](item))
                            logger.info(f"Ingested {json_key} (item: {item}) for journalist {json_data['name']}")
                        except sqlite3.IntegrityError:
                            logger.warning(f"Skipping duplicate {json_key} for {json_data['name']}: {item}")
                        except sqlite3.Error as e:
                            logger.error(f"Error ingesting {json_key} for {json_data['name']} (item: {item}): {e}")
        else:
            logger.debug(f"No {json_key} data found for journalist {json_data['name']}")

    response_format_data = json_data.get('response_format', {})
    response_format_id = None
    if response_format_data and response_format_data.get('structure'):
        try:
            cursor.execute("SELECT id, structure FROM journalist_response_formats WHERE journalist_id = ?", (db_journalist_id,))
            existing_rf = cursor.fetchone()

            if existing_rf:
                response_format_id = existing_rf[0]
                if existing_rf[1] != response_format_data.get('structure'):
                    cursor.execute(
                        "UPDATE journalist_response_formats SET structure = ? WHERE id = ?",
                        (response_format_data.get('structure'), response_format_id)
                    )
                    logger.info(f"Updated response_format for journalist {json_data['name']}")
                else:
                    logger.info(f"Response_format for journalist {json_data['name']} unchanged.")
            else:
                cursor.execute(
                    "INSERT INTO journalist_response_formats (journalist_id, structure) VALUES (?, ?)",
                    (db_journalist_id, response_format_data.get('structure'))
                )
                response_format_id = cursor.lastrowid
                logger.info(f"Ingested new response_format for journalist {json_data['name']}")

            if response_format_id:
                cursor.execute("DELETE FROM journalist_response_format_sections WHERE response_format_id = ?", (response_format_id,))
                logger.info(f"Cleared existing response format sections for {json_data['name']}.")

                sections = response_format_data.get('sections', [])
                for i, section_name in enumerate(sections):
                    try:
                        cursor.execute(
                            "INSERT INTO journalist_response_format_sections (response_format_id, section_name, section_order) VALUES (?, ?, ?)",
                            (response_format_id, section_name, i)
                        )
                        logger.info(f"Ingested response_format_section '{section_name}' for journalist {json_data['name']}")
                    except sqlite3.IntegrityError:
                        logger.warning(f"Skipping duplicate response format section for {json_data['name']}: {section_name}")
                    except sqlite3.Error as e:
                        logger.error(f"Error ingesting response_format_section '{section_name}' for {json_data['name']}: {e}")
        except sqlite3.Error as e:
            logger.error(f"Error ingesting response_format for {json_data['name']}: {e}")
    else:
        logger.debug(f"No response_format data found for journalist {json_data['name']}")

def ingest_data(persona_dir=PERSONA_DIR, corpus_dir=CORPUS_DIR, json_filter_list=None):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        processed_journalists = []

        print(f"\n--- Starting Persona Data Ingestion from {persona_dir} ---")
        for filename in os.listdir(persona_dir):
            if filename.endswith(".json"):
                if json_filter_list and filename not in json_filter_list:
                    print(f"Skipping persona file: {filename} (not in filter list)")
                    logger.info(f"Skipping persona file: {filename} (not in filter list)")
                    continue

                file_path = os.path.join(persona_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        persona_data = json.load(f)

                    journalist_name = persona_data.get('name')
                    if not journalist_name:
                        logger.error(f"Skipping {filename}: 'name' field missing in JSON.")
                        print(f"Error: Skipping {filename} as 'name' field is missing.")
                        continue

                    journalist_id = get_or_create_journalist(
                        conn,
                        journalist_name,
                        persona_data.get('bio', ''),
                        persona_data.get('pic', '')
                    )

                    ingest_persona_data_details(journalist_id, persona_data, cursor, conn)
                    conn.commit()
                    print(f"Successfully ingested persona data for '{journalist_name}'.")
                    logger.info(f"Successfully ingested all persona data for '{journalist_name}'.")
                    processed_journalists.append((journalist_id, journalist_name))

                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from {filename}: {e}")
                    print(f"Error: Invalid JSON in {filename}. Skipping.")
                    conn.rollback()
                except FileNotFoundError:
                    logger.error(f"Persona file not found: {filename}")
                    print(f"Error: Persona file not found: {filename}. Skipping.")
                    conn.rollback()
                except sqlite3.Error as e:
                    logger.error(f"Database error during persona ingestion for {filename}: {e}. Rolling back current transaction.")
                    print(f"Database error for {filename}: {e}. Rolling back.")
                    conn.rollback()
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing {filename}: {e}. Rolling back current transaction.")
                    print(f"An unexpected error occurred while processing {filename}: {e}. Skipping and rolling back.")
                    conn.rollback()

        print(f"\n--- Starting Corpus Data Ingestion from {corpus_dir} ---")
        for j_id, j_name in processed_journalists:
            persona_corpus_dir_name = j_name.lower().replace(" ", "_")
            persona_corpus_path = os.path.join(corpus_dir, persona_corpus_dir_name)
            print(f"Corpus path : {persona_corpus_path}")
            if not os.path.isdir(persona_corpus_path):
                logger.warning(f"Corpus directory for '{j_name}' not found at {persona_corpus_path}. Skipping corpus ingestion for this journalist.")
                print(f"Warning: Corpus directory for '{j_name}' not found. Skipping corpus ingestion.")
                continue

            print(f"Ingesting corpus for journalist: {j_name} (ID: {j_id}) from {persona_corpus_path}")
            for fname in os.listdir(persona_corpus_path):
                if fname.endswith(".txt"):
                    file_path = os.path.join(persona_corpus_path, fname)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        cursor.execute(
                            "INSERT INTO corpus_documents (journalist_id, content) VALUES (?, ?)",
                            (j_id, content)
                        )
                        conn.commit()
                        logger.info(f"Ingested document {fname} for journalist {j_name}")
                    except FileNotFoundError:
                        logger.error(f"Corpus document not found: {file_path}")
                        print(f"Error: Corpus document not found: {file_path}. Skipping.")
                        conn.rollback()
                    except sqlite3.Error as e:
                        logger.error(f"Error ingesting corpus document {fname} for {j_name}: {e}")
                        print(f"Error: Database error ingesting corpus document {fname} for {j_name}: {e}")
                        conn.rollback()
                    except Exception as e:
                        logger.error(f"An unexpected error occurred while processing corpus document {file_path}: {e}")
                        print(f"An unexpected error occurred while processing corpus document {file_path}: {e}. Skipping and rolling back.")
                        conn.rollback()

        print("\n--- Ingestion complete. ---")
        logger.info("Overall ingestion process complete.")

    except sqlite3.Error as e:
        logger.critical(f"A critical database error occurred during overall ingestion: {e}")
        print(f"Critical Database Error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        logger.critical(f"An unhandled error occurred during overall ingestion: {e}")
        print(f"Unhandled Error during ingestion: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def rollback_db(rollback_type="soft", journalist_name=None):
    if rollback_type == "hard":
        print(f"Performing HARD rollback: Deleting database file at {DB_PATH}...")
        if os.path.exists(DB_PATH):
            try:
                os.remove(DB_PATH)
                logger.info(f"HARD rollback successful: Database file '{DB_PATH}' deleted.")
                print(f"Successfully deleted database file: {DB_PATH}")
            except OSError as e:
                logger.error(f"Error deleting database file '{DB_PATH}' during hard rollback: {e}")
                print(f"Error deleting database file: {e}")
        else:
            logger.info(f"No database file found at {DB_PATH} for hard rollback.")
            print(f"No database file found at {DB_PATH}.")
    elif rollback_type == "soft":
        if not journalist_name:
            print("Error: Journalist name is required for a soft rollback.")
            logger.error("Soft rollback attempted without a journalist_name.")
            return

        print(f"Performing SOFT rollback for journalist: '{journalist_name}'...")
        if not os.path.exists(DB_PATH):
            print(f"Database file not found at {DB_PATH}. Cannot perform soft rollback.")
            logger.warning(f"Database file not found for soft rollback: {DB_PATH}")
            return

        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM journalists WHERE name=?", (journalist_name,))
            journalist_id_row = cursor.fetchone()

            if not journalist_id_row:
                print(f"Journalist '{journalist_name}' not found in the database. No soft rollback performed.")
                logger.info(f"Journalist '{journalist_name}' not found for soft rollback.")
                return

            journalist_id = journalist_id_row[0]

            tables_to_clear = [
                "journalist_tones",
                "journalist_interests",
                "journalist_aversions",
                "journalist_triggers",
                "journalist_evaluation_criteria",
                "journalist_report_format_preferences",
                "journalist_influences",
                "journalist_current_focus",
                "journalist_response_format_sections",
                "journalist_response_formats",
                "corpus_documents"
            ]

            for table in tables_to_clear:
                try:
                    if table == "journalist_response_format_sections":
                        cursor.execute("SELECT id FROM journalist_response_formats WHERE journalist_id = ?", (journalist_id,))
                        response_format_ids = [row[0] for row in cursor.fetchall()]
                        if response_format_ids:
                            placeholders = ','.join('?' for _ in response_format_ids)
                            cursor.execute(f"DELETE FROM {table} WHERE response_format_id IN ({placeholders})", response_format_ids)
                            logger.info(f"Deleted {cursor.rowcount} entries from {table} for journalist ID {journalist_id} via response_format_id.")
                        else:
                            logger.info(f"No response formats found for journalist ID {journalist_id}, skipping {table} deletion.")
                    else:
                        cursor.execute(f"DELETE FROM {table} WHERE journalist_id = ?", (journalist_id,))
                        logger.info(f"Deleted {cursor.rowcount} entries from {table} for journalist ID {journalist_id}.")
                    print(f"  Deleted data from '{table}' for '{journalist_name}'.")
                except sqlite3.Error as e:
                    logger.error(f"Error deleting from table '{table}' during soft rollback for journalist '{journalist_name}': {e}")
                    print(f"  Error deleting from '{table}': {e}")
                    conn.rollback()
                    return

            cursor.execute("DELETE FROM journalists WHERE id=?", (journalist_id,))
            conn.commit()
            print(f"Successfully performed SOFT rollback: Deleted all data for journalist '{journalist_name}'.")
            logger.info(f"Successfully performed SOFT rollback for journalist '{journalist_name}' (ID: {journalist_id}).")

        except sqlite3.Error as e:
            logger.error(f"Database error during soft rollback for journalist '{journalist_name}': {e}")
            print(f"Database error during soft rollback: {e}")
            if conn:
                conn.rollback()
        except Exception as e:
            logger.error(f"An unexpected error occurred during soft rollback for journalist '{journalist_name}': {e}")
            print(f"An unexpected error occurred during soft rollback: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    else:
        print("Invalid rollback type. Use 'soft' or 'hard'.")
        logger.warning(f"Invalid rollback type requested: {rollback_type}")

def setup_persona(persona = []):
    init_db()
    ingest_data(
        persona_dir = PERSONA_DIR,
        corpus_dir = CORPUS_DIR,
        json_filter_list = persona
    )

def rollback_persona(persona = []):
    print("\n--- Performing SOFT Rollback for 'Journalist' ---")
    rollback_db(rollback_type="soft", journalist_name="Journalist")
    print("\n--- Re-ingesting 'Journalist A' after soft rollback ---")
    ingest_data(
        persona_dir = PERSONA_DIR,
        corpus_dir = CORPUS_DIR,
        json_filter_list = persona
    )

def recreate_persona(persona = []): # WILL DELETE DB and Create only selected
    rollback_db(rollback_type="hard")
    init_db(recreate=True)
    ingest_data(
        persona_dir=PERSONA_DIR,
        corpus_dir=CORPUS_DIR,
        json_filter_list = persona
    )


if __name__ == "__main__":
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    # setup_persona(["journalist_A.json", "journalist_C.json"])
    # rollback_persona(["journalist_A.json"])
    recreate_persona(["casey.json", "dave.json", "morgan.json"])
    