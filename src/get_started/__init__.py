from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

def main() -> None:
    if API_KEY:
        print(f"Hello from agent-development-playground!")
    else:
        print("Fix your code!!!")
