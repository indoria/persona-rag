from patch.sqlite3 import sqlite3
import os
from dotenv import load_dotenv
from uuid import uuid4
from typing import List
from lib.chunk.tiktoken_chunker import chunk_text
from db.chroma_embedders import SentenceTransformerEmbedder, OpenAIEmbedder
from lib.embed.embedders.azureOpenAIEmbedder import AzureOpenAIEmbedder

load_dotenv()

DB_PATH = os.getenv('DB_PATH')
CHROMA_PATH = os.getenv('CHROMA_PATH')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
EMBEDDING_MODEL_TYPE = os.getenv('EMBEDDING_MODEL_TYPE', 'sbert')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')


def add_text_to_chroma(text: str):
    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(uuid4())]  # Unique ID per chunk
        )



def embed_all_corpus_documents():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, journalist_id, content FROM corpus_documents")
    rows = c.fetchall()
    for doc_id, journalist_id, content in rows:
        # store_embedding(doc_id, journalist_id, content)
        store_embedding_to_chroma(content, doc_id, journalist_id)
    conn.close()
    print(f"Embedded {len(rows)} documents into ChromaDB.")

def get_embedder_instance():
    if EMBEDDING_MODEL_TYPE.lower() == 'sbert':
        if not EMBEDDING_MODEL_NAME:
            raise ValueError("EMBEDDING_MODEL_NAME must be set for 'sbert' model type in .env")
        return SentenceTransformerEmbedder(
            model_name=EMBEDDING_MODEL_NAME,
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )
    elif EMBEDDING_MODEL_TYPE.lower() == 'openai':
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in .env for 'openai' model type.")
        if not EMBEDDING_MODEL_NAME:
            default_openai_model = "text-embedding-ada-002"
            print(f"EMBEDDING_MODEL_NAME not set for OpenAI, defaulting to '{default_openai_model}'.")
            EMBEDDING_MODEL_NAME = default_openai_model
        return OpenAIEmbedder(
            model_name=EMBEDDING_MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )
    else:
        raise ValueError(f"Unsupported EMBEDDING_MODEL_TYPE: '{EMBEDDING_MODEL_TYPE}'. "
                         "Please set it to 'sbert' or 'openai' in your .env file.")

def embed_all_corpus_documents():
    if not DB_PATH:
        print("Error: DB_PATH environment variable is not set. Please set it in your .env file.")
        return

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    embedder = None
    conn = None
    try:
        embedder = get_embedder_instance()

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, journalist_id, content FROM corpus_documents")
        rows = c.fetchall()

        if not rows:
            print("No documents found in 'corpus_documents' table SQLite database.")
            print("Please ensure database is populated.")
            return

        print(f"Found {len(rows)} documents in SQLite database to embed.")
        for doc_id, journalist_id, content in rows:
            embedder.store_embedding(str(doc_id), str(journalist_id), content)

        print(f"Embedding process completed using {EMBEDDING_MODEL_TYPE} model: {EMBEDDING_MODEL_NAME}.")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except RuntimeError as e:
        print(f"Embedding Model Error: {e}")
    except sqlite3.Error as e:
        print(f"SQLite Database Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("SQLite database connection closed.")

def azureEmbedder():
    deployment_name = os.getenv('AZURE_AI_DEPLOYMENT_NAME', 'text-embedding-ada-002')

    embedder = AzureOpenAIEmbedder(deployment_name=deployment_name)
    texts = ["hello world"]
    embedding: List[List[float]] = embedder._encode(texts)

    print("Embedding for 'hello world':")
    print(embedding[0])

def store_embedding_to_chroma(text: str, doc_id: str, journalist_id: str):
    chunks = chunk_text(text)
    # embedder = AzureOpenAIEmbedder(deployment_name=deployment_name)
    for chunk in chunks:
        # embedding: List[List[float]] = embedder._encode([chunk])
        # collection.add(
        #     documents=[chunk],
        #     embeddings=[embedding[0]],
        #     ids=[str(uuid4())]  # Unique ID per chunk
        # )
        print(chunk)
        return

if __name__ == "__main__":
    #embed_all_corpus_documents()
    #azureEmbedder()
    embed_all_corpus_documents()