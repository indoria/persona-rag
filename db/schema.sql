CREATE TABLE IF NOT EXISTS journalists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    bio TEXT
);

CREATE TABLE IF NOT EXISTS journalist_interests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER,
    interest_topic TEXT,
    strength_score REAL,
    FOREIGN KEY (journalist_id) REFERENCES journalists(id)
);

CREATE TABLE IF NOT EXISTS corpus_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER,
    document_type TEXT,
    content TEXT,
    processed_content TEXT,
    avg_sentence_length REAL,
    sentiment_score REAL,
    FOREIGN KEY (journalist_id) REFERENCES journalists(id)
);