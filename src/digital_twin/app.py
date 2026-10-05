import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

from digital_twin.tools import TOOLS, handle_tool_calls

load_dotenv()
PROFILE_PATH = Path(__file__).parent / "me" / "about_me.pdf"
API_KEY = os.environ.get("OPENROUTER_API_KEY")
CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

MODEL = "anthropic/claude-haiku-4.5"


def load_profile() -> str:
    reader = PdfReader(PROFILE_PATH)
    about_me_content = ("\n\n").join(page.extract_text() for page in reader.pages)

    return about_me_content


def build_system_prompt(profile_content: str) -> str:
    prompt = f"""
    You are a digital twin of Ankit, placed on his personal website to engage and chat with visitors.
    visitors will ask questions about him and make business enquiries. 
    You are supposed to answer in a professional, first person tone.
    All the information about him is provided in the profile_data section below.
    DO NOT fabricate any information, always stick to the content in the profile_data section.
    If you don't know something, politely say so.
    
    When a visitor shows interest (in hiring, a collaboration or a follow-up), ask for their email. When they give it, call the tool, and confirm to them that you'll be in touch. Never ask twice.
    
    <profile_data>
    {profile_content}
    </profile_data>
    """

    return prompt


SYSTEM_PROMPT = build_system_prompt(load_profile())


def chat(message: str, history: list[dict]) -> str:
    refined_history = [{"role": m["role"], "content": m["content"]} for m in history]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *refined_history,
        {"role": "user", "content": message},
    ]
    max_turns = 5

    for _ in range(max_turns):
        response = CLIENT.chat.completions.create(
            model=MODEL, max_tokens=1024, messages=messages, tools=TOOLS
        )
        llm_response = response.choices[0].message
        if not llm_response.tool_calls:
            return llm_response.content or ""

        tool_result_messages = handle_tool_calls(llm_response.tool_calls)

        messages.append(llm_response.model_dump(exclude_none=True))
        messages.extend(tool_result_messages)

    return "Something went wrong"


def main() -> None:
    gr.ChatInterface(fn=chat).launch()


if __name__ == "__main__":
    main()
