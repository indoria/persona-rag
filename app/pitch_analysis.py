import spacy

nlp = spacy.load("en_core_web_sm")

def analyze_pitch(pitch_text):
    """
    Analyze the pitch, extracting entities and key noun phrases.
    Returns:
        {
            "entities": [(text, label), ...],
            "noun_chunks": [chunk, ...],
        }
    """
    doc = nlp(pitch_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    return {
        "entities": entities,
        "noun_chunks": noun_chunks,
    }