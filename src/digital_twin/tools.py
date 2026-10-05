import json
from datetime import UTC, datetime
from pathlib import Path

from openai.types.chat import ChatCompletionMessageToolCallUnion

LEADS_PATH = Path(__file__).parent / "me" / "leads.jsonl"

def record_lead_details(email:str, name: str | None = None, note: str | None = None) -> str:
    """Records lead details - email, note and their name. Note and name can be optional"""
    
    formatted_email = email.strip()
    if "@" not in formatted_email:
        return "Email ID is in invalid format, ask the user to provide valid one"
    
    record = {
        "email": formatted_email,
        "name": name,
        "note": note,
        "timestamp": datetime.now(UTC).isoformat()
    }
    
    with open(LEADS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    
    return "Successfully recorded"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_lead_details",
            "description": (
                "Record a visitor's contact details when they share their email address, "
                "e.g. for hiring, collaboration or a follow-up conversation. "
                "Only call this once the visitor has explicitly provided an email."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The visitor's email address, exactly as they provided it.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The visitor's name, if they mentioned it.",
                    },
                    "note": {
                        "type": "string",
                        "description": "One-line summary of what the visitor is interested in or why they reached out.",
                    },
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_FUNCTIONS = {"record_lead_details": record_lead_details}


def handle_tool_calls(tool_calls: list[ChatCompletionMessageToolCallUnion]) -> list[dict]:
    """Execute the model's requested tool calls and return the tool result messages."""
    results = []
    for call in tool_calls:
        if call.type != "function":
            continue

        fn = TOOL_FUNCTIONS.get(call.function.name)
        if fn is None:
            result = f"Unknown tool: {call.function.name}"
        else:
            try:
                args = json.loads(call.function.arguments)
                result = fn(**args)
            except (json.JSONDecodeError, TypeError) as e:
                result = f"Error calling tool: {e}"

        print(f"[tool] {call.function.name}({call.function.arguments}) -> {result}")
        results.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return results


if __name__== "__main__":
    print(record_lead_details("ankit@abc.com", "Ankit", "very good"))
    print(record_lead_details("ankit2@abc.com", "Ankit"))
    print(record_lead_details("ankitabc.com", "Ankit", "very good"))