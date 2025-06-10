import os
import pytest

def test_process_text_valid():
    from data_ingestion.text_processing import process_text
    txt = "Tesla builds electric cars in California."
    result = process_text(txt)
    # Check that Tesla and California are recognized entities
    entities = [ent[0] for ent in result["entities"]]
    assert "Tesla" in entities
    assert "California" in entities
    assert len(result["clean_tokens"]) > 0

def test_extract_features_sentiment():
    from data_ingestion.feature_extraction import extract_features
    processed = {"lemmas": ["great", "innovation", "in", "renewable", "energy"]}
    feats = extract_features(processed)
    # Sentiment should be nonzero for positive words
    assert "sentiment" in feats
    assert isinstance(feats["sentiment"], float)

def test_extract_style_features():
    from data_ingestion.feature_extraction import extract_style_features
    text = "Amazing progress! Solar panels are everywhere. Clean energy is the future."
    style = extract_style_features(text)
    assert style["avg_sentence_length"] > 0
    assert style["vocab_richness"] > 0