import spacy

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    lemmas = []
    entities = []
    pos_tags = []
    clean_tokens = []
    for token in doc:
        if not (token.is_stop or token.is_punct or token.is_space):
            clean_tokens.append(token.text)
            lemmas.append(token.lemma_)
            pos_tags.append(token.pos_)
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return {
        "lemmas": lemmas,
        "entities": entities,
        "pos_tags": pos_tags,
        "clean_tokens": clean_tokens,
        "doc": doc,
    }