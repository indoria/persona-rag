# PR Journalist AI Persona POC

## How to Run (Phase 3)

1. **Install dependencies**  
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Ingest corpus and fine-tune persona models**  
   ```
   python data_ingestion/ingest.py (or python -m data_ingestion.ingest)
   python -m data_ingestion/embed_corpus
   python -m scripts.fine_tune_all
   ```

3. **Run the Flask app**  
   ```
   python run.py
   ```
   Visit [http://localhost:8001](http://localhost:8001) in your browser.

4. **Try it!**
   - Select a journalist.
   - Paste your PR pitch.
   - Get an AI-generated journalist persona response!

---

## End-to-End Data Flow

1. **User** selects persona and inputs a pitch on the UI.
2. **Frontend** sends pitch and persona selection to Flask backend.
3. **Backend**
    - Analyzes pitch (entities, topics).
    - Retrieves and loads journalist’s fine-tuned model.
    - Pulls relevant context from ChromaDB.
    - Generates persona-style response.
4. **Frontend** displays the response.

---

## Next Steps

- Add more journalists, or a real corpus.
- Improve LLM fine-tuning or try retrieval-augmented generation.
- Harden error handling and add authentication for production use.
- See `tests/` directory for test scaffolding.


```
scikit learn vs pytorch vs tensorflow vs keras vs gensim
```

```
Updating python (and in turn sqlite3 as well)

curl -fsSL https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
source ~/.bashrc
pyenv update

pyenv install --list
pyenv install <version>
pyenv global <version>
pyenv local <version>
pyenv version
pyenv versions
```