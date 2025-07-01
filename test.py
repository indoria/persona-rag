from app.prompts import load_prompt
from openai import OpenAI, AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_ROUTER_API_KEY = os.getenv("OPENAI_ROUTER_API_KEY")

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

def get_openrouter_response(prompt: str, model: str = "openai/gpt-4") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    
    return response.choices[0].message.content


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


if __name__ == "__main__":
    prompt = load_prompt(
        journalist_key="morgan_housel",
        tone="insight",
        context=[
            "Most people underreact to how compounding works.",
            "Your behavior matters more than your skill when it comes to money."
        ],
        press_release="Vanguard announces a new low-cost fund targeting Gen-Z investors."
    )
    response = get_azure_gpt4_response(prompt)
    print(response)
    # personas = get_available_personas()
    # print(personas)