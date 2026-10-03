import os

from openai import OpenAI
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Fix your code!!! OPENROUTER_API_KEY is not set.")
        return
    
    print("Hello from agent-development-playground!")
    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = api_key
    )
    response = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Say hello"}],
    )
    print(response.choices[0].message.content)
