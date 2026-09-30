import os

import anthropic
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Fix your code!!! ANTHROPIC_API_KEY is not set.")
        return

    print("Hello from agent-development-playground!")
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": "Say hello"}],
    )
    print(msg.content[0].text)
