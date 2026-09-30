from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

client = anthropic.Anthropic()

def main() -> None:
    if API_KEY:
        print(f"Hello from agent-development-playground!")
        msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": "Say hello"}],
            )
        print(msg.content[0].text)
    else:
        print("Fix your code!!!")
