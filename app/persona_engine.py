import logging

logger = logging.getLogger(__name__)

import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch

MODEL_ROOT = "database/models"

def fine_tune_journalist_model(journalist_name, corpus_texts, model_name="distilgpt2", output_dir=None, epochs=1):
    """
    Fine-tune a small GPT-2 model on the journalist's corpus.
    """
    if output_dir is None:
        output_dir = os.path.join(MODEL_ROOT, journalist_name.lower())
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token
    
    # Save corpus to a temporary file
    corpus_path = os.path.join(output_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in corpus_texts:
            f.write(doc.strip() + "\n\n")
    
    # Create dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=corpus_path,
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        save_steps=20,
        save_total_limit=2,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved at {output_dir}")

def load_persona_model(journalist_name):
    """
    Load the fine-tuned model/tokenizer for the given journalist.
    """
    path = os.path.join(MODEL_ROOT, journalist_name.lower())
    model = GPT2LMHeadModel.from_pretrained(path)
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_persona_response(journalist_id, pitch_text, db_conn, chroma_client, num_context=1, max_length=128, temperature=0.8):
    """
    Given journalist_id and pitch_text, generate a persona-driven response.
    Optionally retrieves context examples from ChromaDB.
    """
    # 1. Get journalist name
    c = db_conn.cursor()
    c.execute("SELECT name FROM journalists WHERE id=?", (journalist_id,))
    row = c.fetchone()
    if not row:
        return "Error: Journalist not found."
    journalist_name = row[0]
    # 2. Load model
    model, tokenizer = load_persona_model(journalist_name)
    # 3. Retrieve context from ChromaDB
    context = ""
    try:
        col = chroma_client.get_collection("corpus_embeddings")
        results = col.query(
            query_texts=[pitch_text],
            n_results=num_context,
            where={"journalist_id": journalist_id},
        )
        print(["results", results, "Journalist_id", journalist_id, 'pitch_text', pitch_text])
        for doc in results["documents"]:
            context += doc + "\n"
    except Exception as e:
        print(e)
        return "I do not know much about it."
        pass  # For POC, skip if unavailable
    # 4. Construct prompt
    prompt = ""
    if context:
        prompt += f"[Reference examples]\n{context.strip()}\n\n"
    prompt += f"[User PR pitch]\n{pitch_text.strip()}\n\n[Response]\n"
    # 5. Generate response
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
    )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the new response portion after the [Response] tag
    response = full_text.split("[Response]", 1)[-1].strip()
    return response if response else full_text