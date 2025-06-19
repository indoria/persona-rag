import spacy
from spacy.symbols import nsubj, VERB
from collections import Counter
from heapq import nlargest

# https://spacy.io/models
# nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def spacyNLP(pitch_text):
    doc = nlp(pitch_text)
    results = {}

    results["text"] = doc.text
    results["number_of_tokens"] = len(doc)
    results["named_entities"] = [(ent.text, ent.label_) for ent in doc.ents]
    results["noun_chunks"] = [(chunk.text, chunk.root) for chunk in doc.noun_chunks]
    results["sentences"] = doc.sents
    results["dir_doc"] = dir(doc)

    doc_properties_methods = {}
    for attr in dir(doc):
        if not attr.startswith('_'):
            try:
                value = getattr(doc, attr)
                if callable(value):
                    doc_properties_methods[attr] = "Method"
                else:
                    doc_properties_methods[attr] = f"Property: {value}"
            except Exception as e:
                doc_properties_methods[attr] = f"Error accessing: {e}"
    results["doc_properties_methods"] = doc_properties_methods

    # token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop
    token_properties = []
    for token in doc:
        partOfSpeech = {
            "text": token.text,
            "pos": token.pos_,
            "lemma": token.lemma_,
            "is_stop": token.is_stop,
            "tag": token.tag_,
            "dep": token.dep_,
            "shape": token.shape_,
            "is_alpha": token.is_alpha
        }

        morphology = {
            "morphology": token.morph
        }
        token_properties.append(**partOfSpeech, **morphology)

    results["token_properties"] = token_properties

    verbs = set()
    for possible_subject in doc:
        if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
            verbs.add(possible_subject.head)
    results["verbs"] = verbs

    return results

def pitch_nlp(pitch_text):
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

def _calculate_word_frequencies(doc):
    """Calculates normalized word frequencies for keywords."""
    keywords = [
        token.text.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    word_frequencies = Counter(keywords)
    max_frequency = max(word_frequencies.values()) if word_frequencies else 1
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency
    return word_frequencies

def _score_sentences(doc, word_frequencies, entity_boost_factor=1.5):
    """Scores sentences based on word frequencies and entity presence."""
    sentence_scores = {}
    for sent in doc.sents:
        score = sum(word_frequencies.get(word.text.lower(), 0) for word in sent)
        if sent.ents:
            score *= entity_boost_factor
        
        # Use hash or something else for true uniqueness of the key
        sentence_scores[sent.text] = score
    return sentence_scores

def _reconstruct_summary_order(original_sentences, summarized_sentences_text):
    """Reconstructs the summary in the original sentence order."""
    final_summary_sentences = []
    # Use a set for faster lookup of summarized sentences
    summarized_set = set(summarized_sentences_text)
    for original_sent in original_sentences:
        if original_sent in summarized_set:
            final_summary_sentences.append(original_sent)
            summarized_set.remove(original_sent) # Ensure unique inclusion if text identical
    return " ".join(final_summary_sentences)

def summarize(text, num_sentences=3):
    """
    Summarizes the input text using keyword frequency and entity recognition.

    Args:
        text (str): The input text to summarize.
        num_sentences (int): The number of top sentences to include in the summary.

    Returns:
        str: The summarized text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if not isinstance(num_sentences, int) or num_sentences <= 0:
        raise ValueError("num_sentences must be a positive integer.")

    doc = nlp(text)

    if not doc.sents:
        return ""

    word_frequencies = _calculate_word_frequencies(doc)
    sentence_scores = _score_sentences(doc, word_frequencies)

    # Ensure we don't try to get more sentences than available
    num_sentences = min(num_sentences, len(sentence_scores))

    summarized_sentences_text = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    original_sentences = [s.text for s in doc.sents]
    final_summary = _reconstruct_summary_order(original_sentences, summarized_sentences_text)

    return final_summary