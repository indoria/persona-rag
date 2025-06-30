-- Immutable Journalist Information
CREATE TABLE IF NOT EXISTS journalists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL UNIQUE,
    name_first TEXT,
    name_last TEXT,
    name_full TEXT,
    name_display TEXT,
    pic TEXT,
    bio TEXT,
    created_at TEXT,
    updated_at TEXT
);


-- Mutable Contact and Professional Info
CREATE TABLE IF NOT EXISTS journalist_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    email TEXT,
    phone TEXT,
    twitter TEXT,
    linkedin TEXT,
    title TEXT,
    organization_name TEXT,
    organization_department TEXT,
    organization_role TEXT,
    employment_status TEXT CHECK (employment_status IN ('staff', 'freelance', 'contract')),
    latitude REAL,
    longitude REAL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Sentiment Snapshot History
CREATE TABLE IF NOT EXISTS journalist_sentiment_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    polarity REAL,
    subjectivity REAL,
    consistency_score REAL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Crisis Management Snapshot
CREATE TABLE IF NOT EXISTS journalist_crisis_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    crisis_response_score REAL,
    fairness_index REAL,
    responsiveness_score REAL,
    deadline_adherence REAL,
    fact_check_accuracy REAL,
    quote_verification REAL,
    embargo_compliance REAL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Engagement Metrics Snapshot
CREATE TABLE IF NOT EXISTS journalist_engagement_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    response_time_avg REAL,
    relationship_strength REAL,
    email_open_rate REAL,
    response_rate REAL,
    meeting_acceptance_rate REAL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Location hierarchy
CREATE TABLE IF NOT EXISTS journalist_location_hierarchy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    country_code TEXT,
    country_name TEXT,
    region_code TEXT,
    region_name TEXT,
    city_name TEXT,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Coverage areas
CREATE TABLE IF NOT EXISTS journalist_coverage_areas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    type TEXT CHECK (type IN ('primary', 'secondary', 'occasional')),
    geo_bounds TEXT,
    expertise_level TEXT CHECK (expertise_level IN ('novice', 'intermediate', 'expert')),
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Primary beats
CREATE TABLE IF NOT EXISTS journalist_primary_beats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    iptc_code TEXT,
    category TEXT,
    expertise_level TEXT CHECK (expertise_level IN ('novice', 'intermediate', 'expert')),
    years_experience INTEGER,
    confidence_score REAL,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Topic preferences
CREATE TABLE IF NOT EXISTS journalist_topic_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    topic TEXT,
    interest_level INTEGER CHECK (interest_level BETWEEN 1 AND 10),
    coverage_frequency TEXT,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Content analysis
CREATE TABLE IF NOT EXISTS journalist_content_analysis (
    journalist_id INTEGER PRIMARY KEY,
    avg_sentence_length REAL,
    vocabulary_richness REAL,
    formality_score REAL,
    objectivity_score REAL,
    article_frequency REAL,
    word_count_avg INTEGER,
    source_diversity REAL,
    multimedia_usage REAL,
    interview_style TEXT,
    question_complexity REAL,
    follow_up_tendency REAL,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Topic sentiment map
CREATE TABLE IF NOT EXISTS journalist_topic_sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    topic TEXT NOT NULL,
    avg_polarity REAL,
    trend TEXT CHECK (trend IN ('increasing', 'decreasing', 'stable')),
    sample_size INTEGER,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Temporal sentiment patterns
CREATE TABLE IF NOT EXISTS journalist_temporal_sentiment_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    time_period TEXT,
    sentiment_metrics TEXT,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Communication preferences
CREATE TABLE IF NOT EXISTS journalist_communication_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    preferred_channel TEXT,
    optimal_contact_time TEXT,
    preferred_format TEXT,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Interaction history
CREATE TABLE IF NOT EXISTS journalist_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    timestamp TEXT,
    interaction_type TEXT,
    outcome TEXT,
    sentiment REAL,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
);

-- Metadata sources
CREATE TABLE IF NOT EXISTS journalist_metadata_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    data_source TEXT,
    FOREIGN KEY (journalist_id) REFERENCES journalists(journalist_id) ON DELETE CASCADE
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

CREATE TABLE IF NOT EXISTS journalist_triggers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    journalist_id INTEGER NOT NULL,
    trigger_topic TEXT NOT NULL,
    UNIQUE (journalist_id, trigger_topic),
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

-- Indexes for Foreign Key Columns
CREATE INDEX IF NOT EXISTS idx_journalist_profiles_journalist_id ON journalist_profiles(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_sentiment_snapshots_journalist_id ON journalist_sentiment_snapshots(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_crisis_snapshots_journalist_id ON journalist_crisis_snapshots(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_engagement_metrics_journalist_id ON journalist_engagement_metrics(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_location_hierarchy_journalist_id ON journalist_location_hierarchy(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_coverage_areas_journalist_id ON journalist_coverage_areas(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_primary_beats_journalist_id ON journalist_primary_beats(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_topic_preferences_journalist_id ON journalist_topic_preferences(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_topic_sentiments_journalist_id ON journalist_topic_sentiments(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_temporal_sentiment_patterns_journalist_id ON journalist_temporal_sentiment_patterns(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_communication_preferences_journalist_id ON journalist_communication_preferences(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_interactions_journalist_id ON journalist_interactions(journalist_id);
CREATE INDEX IF NOT EXISTS idx_journalist_metadata_sources_journalist_id ON journalist_metadata_sources(journalist_id);
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
