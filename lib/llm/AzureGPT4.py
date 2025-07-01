from app.prompts import load_prompt
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_GPT_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_GPT_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_GPT_NAME")

azure_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2025-01-01-preview",
    base_url=f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_NAME}/"
)

def get_azure_gpt4_response(prompt: str, temperature: float = 0.7) -> str:
    response = azure_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,  # required by SDK but ignored by Azure
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content