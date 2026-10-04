import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
PROFILE_PATH = Path(__file__).parent / "me" / "about_me.pdf"
API_KEY = os.environ.get("OPENROUTER_API_KEY")
CLIENT = OpenAI(
   base_url = "https://openrouter.ai/api/v1",
   api_key = API_KEY
)

MODEL = "anthropic/claude-haiku-4.5"

def load_profile() -> str:
    reader = PdfReader(PROFILE_PATH)
    about_me_content = ("\n\n").join(page.extract_text() for page in reader.pages)
    
    return about_me_content

def build_system_prompt(profile_content:str) -> str:
    prompt = f"""
    You are a digital twin of Ankit, placed on his personal website to engage and chat with visitors.
    visitors will ask questions about him and make business enquiries. 
    You are supposed to answer in a professional, first person tone.
    All the information about him is provided in the profile_data section below.
    DO NOT fabricate any information, always stick to the content in the profile_data section.
    If you don't know something, politely say so.
    
    <profile_data>
    {profile_content}
    </profile_data>
    """
      
    return prompt

SYSTEM_PROMPT = build_system_prompt(load_profile())

def chat(message: str, history: list[dict]) -> str | None:
    refined_history = [{"role": m["role"], "content": m["content"]} for m in history]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                *refined_history,
                {"role": "user", "content": message}
                ]
    response = CLIENT.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=messages
    )
    assistant = response.choices[0].message.content
    
    return assistant

def main() -> None:
    gr.ChatInterface(fn=chat).launch()

if __name__ == "__main__":
    main()
    
    