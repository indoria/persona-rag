from lib.nlp.spacy_nlp import pitch_nlp, summarize

def analyze_pitch(text):
    """
    Analyze the pitch, extracting entities and key noun phrases.
    """
    analysis = pitch_nlp(text)
    summary = summarize(text, 4)
    return { **analysis, "objective_summary": summary }