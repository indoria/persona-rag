PHASE 2: AI Persona Engine Development
--------------------------------------

This phase implements the core generative journalist persona logic:

- Fine-tune a small LLM (e.g., distilgpt2) on each journalist's corpus.
- Store model weights and tokenizer per persona.
- Provide a function to generate responses from a selected persona, using both the pitch and relevant style/context.

You will find:
- Model fine-tuning script (one-off, per-journalist).
- Persona generation module with `generate_persona_response`.
- Model save/load logic.