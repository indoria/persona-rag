from textblob import TextBlob
from collections import Counter
import numpy as np
from .lda_topic_model import topic_model, get_topic_terms

def extract_features(processed):
    # Sentiment (TextBlob, simple for POC)
    text = " ".join(processed["lemmas"])
    sentiment = TextBlob(text).sentiment.polarity
    return {"sentiment": sentiment}

def extract_style_features(raw_text):
    blob = TextBlob(raw_text)
    sentences = blob.sentences
    words = blob.words
    avg_sentence_length = np.mean([len(s.words) for s in sentences]) if sentences else 0
    vocab_richness = len(set(words)) / (len(words) + 1e-6)
    adverbs = sum(1 for w, p in blob.tags if p == "RB")
    adjectives = sum(1 for w, p in blob.tags if p == "JJ")
    return {
        "avg_sentence_length": avg_sentence_length,
        "vocab_richness": vocab_richness,
        "adverb_freq": adverbs / (len(words) + 1e-6),
        "adjective_freq": adjectives / (len(words) + 1e-6),
    }

def extract_topics(lemmas, n_topics=2):
    # For POC, use shared topic model (see lda_topic_model)
    doc = " ".join(lemmas)
    topics = topic_model([doc])
    # Map topic index to top term as interest
    main_topics = {}
    for idx, score in enumerate(topics[0]):
        if score > 0.1:
            main_topics[get_topic_terms(idx)] = float(score)
    return main_topics