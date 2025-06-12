CREATE TABLE IF NOT EXISTS journalists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    bio TEXT,
    pic TEXT,
    role TEXT
);

CREATE TABLE IF NOT EXISTS journalist_tones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL UNIQUE,
    voice TEXT NOT NULL,
    style TEXT NOT NULL,
    formality TEXT NOT NULL,
    FOREIGN KEY (journalist_id) REFERENCES journalists(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS journalist_interests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    interest_topic TEXT NOT NULL,
    UNIQUE (journalist_id, interest_topic),
    FOREIGN KEY (journalist_id) REFERENCES journalists(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS journalist_aversions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    aversion_topic TEXT NOT NULL,
    UNIQUE (journalist_id, aversion_topic),
    FOREIGN KEY (journalist_id) REFERENCES journalists(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS journalist_evaluation_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    criterion_name TEXT NOT NULL,
    question TEXT NOT NULL,
    weight REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    UNIQUE (journalist_id, criterion_name),
    FOREIGN KEY (journalist_id) REFERENCES journalists(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS journalist_report_format_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    format_name TEXT NOT NULL,
    UNIQUE (journalist_id, format_name),
    FOREIGN KEY (journalist_id) REFERENCES journalists(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS journalist_influences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    influence_description TEXT NOT NULL,
    UNIQUE (journalist_id, influence_description),
    FOREIGN KEY (journalist_id) REFERENCES journalists(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS journalist_current_focus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    focus_topic TEXT NOT NULL,
    UNIQUE (journalist_id, focus_topic),
    FOREIGN KEY (journalist_id) REFERENCES journalists(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS journalist_response_formats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL UNIQUE,
    structure TEXT NOT NULL,
    FOREIGN KEY (journalist_id) REFERENCES journalists(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS journalist_response_format_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_format_id INTEGER NOT NULL,
    section_name TEXT NOT NULL,
    section_order INTEGER NOT NULL,
    UNIQUE (response_format_id, section_name),
    FOREIGN KEY (response_format_id) REFERENCES journalist_response_formats(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS corpus_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    document_type TEXT,
    content TEXT,
    processed_content TEXT,
    avg_sentence_length REAL,
    sentiment_score REAL,
    FOREIGN KEY (journalist_id) REFERENCES journalists(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_journalist_tones_journalist_id ON journalist_tones(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_interests_journalist_id ON journalist_interests(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_aversions_journalist_id ON journalist_aversions(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_eval_criteria_journalist_id ON journalist_evaluation_criteria(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_report_formats_journalist_id ON journalist_report_format_preferences(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_influences_journalist_id ON journalist_influences(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_current_focus_journalist_id ON journalist_current_focus(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_response_formats_journalist_id ON journalist_response_formats(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_response_format_sections_format_id ON journalist_response_format_sections(response_format_id);
CREATE INDEX IF NOT EXISTS idx_corpus_documents_journalist_id ON corpus_documents(journalist_id);
