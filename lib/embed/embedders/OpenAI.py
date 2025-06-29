import openai
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
#openai.base_url = "https://openrouter.ai/api/v1"

def get_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]:
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def get_embeddings(text: str, model: str = "text-embedding-3-large") -> List[float]:
    from openai import OpenAI
    client = OpenAI()

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding
