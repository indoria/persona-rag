def test_analyze_pitch_entities():
    from app.pitch_analysis import analyze_pitch
    pitch = "Pitch for HealthAI, a startup using AI to improve diagnostics."
    result = analyze_pitch(pitch)
    # 'HealthAI' and 'AI' should be extracted as entities or noun chunks
    assert any("HealthAI" in ent for ent, _ in result["entities"]) or any("HealthAI" in chunk for chunk in result["noun_chunks"])

def test_analyze_pitch_noun_chunks():
    from app.pitch_analysis import analyze_pitch
    pitch = "Our new product, EcoCar, is an electric vehicle for urban commuters."
    result = analyze_pitch(pitch)
    assert "electric vehicle" in result["noun_chunks"] or "urban commuters" in result["noun_chunks"]