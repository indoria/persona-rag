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
   python -m data_ingestion.ingest (or python data_ingestion/ingest.py)
   python -m data_ingestion.embed_corpus
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

## Sample press releases
[Sample 1](https://prlab.co/blog/examples-of-press-release-by-type/)
[Sample 2, and classification](https://channelvmedia.com/blog/press-release-examples-by-type/)
[Tips](https://www.contentgrip.com/how-to-write-a-press-release-examples/)


## NLP : Spacy en_core_web_sm vs en_core_web_lg
```
The en_core_web_lg (788 MB) compared to en_core_web_sm (10 MB): Around 79 times bigger, and hence a bit slower to load

LAS: 90.07% vs 89.66%
POS: 96.98% vs 96.78%
UAS: 91.83% vs 91.53%
NER F-score: 86.62% vs 85.86%
NER precision: 87.03% vs 86.33%
NER recall: 86.20% vs 85.39%
All that while en_core_web_lg is 79 times larger, hence loads a lot more slowly.
```