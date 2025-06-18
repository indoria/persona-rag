import spacy

# https://spacy.io/models
# nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])

nlp = spacy.load("en_core_web_sm")

pitch_text = "Barkha Dutt is a renowned journalist."
doc = nlp(pitch_text)

print("Text:", doc.text)
print("Number of tokens:", len(doc))
print("Named Entities:", [(ent.text, ent.label_) for ent in doc.ents])
print("Noun Chunks:", [chunk.text for chunk in doc.noun_chunks])

print("\nProperties and Methods of 'doc':\n")
for attr in dir(doc):
    # Filter out private and special attributes
    if not attr.startswith('_'):
        try:
            # Get the attribute value
            value = getattr(doc, attr)
            # Check if it's callable (a method)
            if callable(value):
                print(f"{attr}() - Method")
            else:
                print(f"{attr} - Property: {value}")
        except Exception as e:
            print(f"{attr} - Error accessing: {e}")

print("\nToken Properties:")
for token in doc:
    print(f"{token.text} - POS: {token.pos_}, Lemma: {token.lemma_}, Is Stop: {token.is_stop}")
