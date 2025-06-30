PHASE 3: Integrated Flask Backend, Pitch Analysis, and Minimal Frontend

This phase brings together:
- The Flask backend API with endpoints for journalist listing and persona response generation.
- The pitch analysis module (spaCy-based) for extracting key entities and topics.
- A simple web frontend for user interaction.
- Integration with the SQLite DB and ChromaDB, and model serving logic.

Folder highlights:
- `app/api.py`: Flask app with required endpoints.
- `app/pitch_analysis.py`: spaCy-based pitch analyzer.
- `frontend/index.html`, `frontend/main.js`, `frontend/style.css`: Minimal interactive UI.

You’re ready for end-to-end testing after this phase.