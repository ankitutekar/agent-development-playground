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
    DO NOT fabricate any information.

    Rules:
    - Answer only from profile_data. Do not use outside or general knowledge, even if you know the answer.
    - If a question isn't answered by profile_data (including off-topic or general-knowledge questions),
      first call record_unknown_question with the visitor's question, then politely say you don't know
      and steer the conversation back to Ankit's work.
    - When a visitor shows interest (hiring, collaboration or a follow-up), ask for their email once.
      When they share it, call record_lead_details, then confirm that you'll be in touch.
    
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
        if not llm_response.tool_calls:  # or could use finish_reason
            return llm_response.content or ""

        tool_result_messages = handle_tool_calls(llm_response.tool_calls)

        messages.append(llm_response.model_dump(exclude_none=True))
        messages.extend(tool_result_messages)

    return "Something went wrong"


TITLE = "🌊 Chat with Ankit"

DESCRIPTION = """
Hi, I'm Ankit's **digital twin**: a backend engineer from Pune working with .NET, Azure and agentic AI.
Ask me about my experience, skills or projects, or leave your email if you'd like to work together.
"""

EXAMPLES = [
    "Tell me about your career journey so far.",
    "What's your core tech stack?",
    "What have you been building with LLMs and agents?",
    "I'd like to discuss a role with you. How can we connect?",
]

CSS = """
.gradio-container { max-width: 860px !important; margin: 0 auto !important; }
footer { display: none !important; }
"""


def main() -> None:
    chatbot = gr.Chatbot(
        show_label=False,
        scale=1,
        placeholder="**Hi there! 👋** Pick an example below or type your own question.",
    )
    textbox = gr.Textbox(placeholder="Ask me anything about my work…", scale=7)

    demo = gr.ChatInterface(
        fn=chat,
        chatbot=chatbot,
        textbox=textbox,
        title=TITLE,
        description=DESCRIPTION,
        examples=EXAMPLES,
        cache_examples=False,
    )
    demo.launch(theme=gr.themes.Ocean(), css=CSS)


if __name__ == "__main__":
    main()
