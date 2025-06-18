# Phase 4: Testing & Debugging

This phase ensures your PR Journalist AI Persona POC is robust, reliable, and ready for demos or early user feedback. It covers **unit tests**, **integration tests**, and **debugging/logging setup**.

---

## 1. **Unit Tests**

Write focused unit tests for all critical modules:

### a. Data Ingestion

- **Test file reading/parsing** (e.g., verify corpus files load and parse correctly).
- **Test text processing** (tokenization, lemmatization, entity extraction).
- **Test feature and topic extraction** (verify correct metrics and topic inference).

### b. Pitch Analysis

- **Test `analyze_pitch` function** on a variety of pitch texts:
  - Correct entity extraction
  - Reasonable noun chunk extraction

### c. Persona Generation

- **Test `generate_persona_response`**:
  - Does it return a non-empty string for valid input?
  - Does it handle missing/nonexistent journalist IDs gracefully?

---

## 2. **Integration Testing**

Test the pipeline end-to-end:

- **UI → Flask → NLP → Persona Engine → Flask → UI**
  - Simulate API calls with predefined pitches and persona selections.
  - Check that the full response cycle completes and the output is displayed.
- Use tools like [pytest](https://docs.pytest.org/en/stable/), [Flask's test client](https://flask.palletsprojects.com/en/2.3.x/testing/), or [requests](https://docs.python-requests.org/en/latest/) for API tests.

---

## 3. **Debugging & Logging**

- **Add logging** to Flask endpoints and key Python modules:
  - Log errors, exceptions, and important state transitions.
  - Use Python’s built-in `logging` module.
- **Set Flask to debug mode** in development, but turn off for production.
- **Catch and log exceptions** in persona generation, DB access, and ChromaDB calls.

---

## 4. **Sample Test Scaffolding**

Place tests in the `tests/` folder:

```python name=tests/test_data_ingestion.py
def test_process_text_valid():
    from data_ingestion.text_processing import process_text
    txt = "Tesla builds electric cars in California."
    result = process_text(txt)
    assert "Tesla" in [ent[0] for ent in result["entities"]]
    assert len(result["clean_tokens"]) > 0
```

```python name=tests/test_pitch_analysis.py
def test_analyze_pitch_entities():
    from app.pitch_analysis import analyze_pitch
    pitch = "Pitch for HealthAI, a startup using AI to improve diagnostics."
    result = analyze_pitch(pitch)
    assert any("HealthAI" in ent for ent, _ in result["entities"])
```

```python name=tests/test_persona_engine.py
def test_generate_persona_response_smoke():
    from app.persona_engine import generate_persona_response
    from patch.sqlite3 import sqlite3, chromadb
    db_conn = sqlite3.connect("persona.db")
    chroma_client = chromadb.PersistentClient("pr_journalist_chroma")
    # Use a valid journalist_id from your DB
    resp = generate_persona_response(1, "Tell me about Tesla's new battery.", db_conn, chroma_client)
    assert isinstance(resp, str) and len(resp) > 0
```

---

## 5. **Manual Testing & Debugging Tips**

- Try the UI with various pitches and personas.
- Use logging output to trace failures.
- Try edge cases: empty pitch, unknown journalist, very long pitch.

---

## 6. **Next Steps (Post-POC)**

- Expand automated test coverage.
- Add CI/CD for automated test runs.
- Collect user feedback and iterate.

---

**Summary Table for Phase 4 Tasks**

| Task Area           | Actions                                               | Location         |
|---------------------|-------------------------------------------------------|------------------|
| Unit Testing        | Write tests for ingestion, NLP, persona gen           | `tests/`         |
| Integration Testing | Test API endpoints and UI end-to-end                  | `tests/` or tools|
| Logging             | Add error/info logs throughout backend and API        | codebase         |
| Manual Debugging    | Try different user flows and edge cases in UI         | UI/API           |

---