from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle

# Fit topic model on all corpus docs on first run; else, load it
MODEL_PATH = "data_ingestion/topic_model.pkl"
TERM_PATH = "data_ingestion/topic_terms.pkl"

def train_topic_model(corpus_texts, n_topics=5):
    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(corpus_texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    with open(MODEL_PATH, "wb") as f: pickle.dump(lda, f)
    with open(TERM_PATH, "wb") as f: pickle.dump(vectorizer.get_feature_names_out(), f)
    return lda, vectorizer

def load_topic_model():
    with open(MODEL_PATH, "rb") as f: lda = pickle.load(f)
    with open(TERM_PATH, "rb") as f: terms = pickle.load(f)
    return lda, terms

def topic_model(texts):
    # Fit or load
    if not os.path.exists(MODEL_PATH):
        lda, vectorizer = train_topic_model(texts)
    else:
        lda, terms = load_topic_model()
        vectorizer = CountVectorizer(vocabulary=terms)
    X = vectorizer.transform(texts)
    return lda.transform(X)

def get_topic_terms(topic_idx, top_n=2):
    lda, terms = load_topic_model()
    topic = lda.components_[topic_idx]
    top_terms = [terms[i] for i in topic.argsort()[-top_n:][::-1]]
    return ", ".join(top_terms)