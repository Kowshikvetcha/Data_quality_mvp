from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from openai import OpenAI
#from Data_quality_mvp.core.ai_tools import CLEANING_TOOLS
from core.ai_tools import CLEANING_TOOLS
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_system_prompt(column_types: dict) -> str:
    cols = "\n".join([f"- {c} ({t})" for c, t in column_types.items()])
    return f"""
You are a data cleaning assistant.

Available columns:
{cols}

Rules:
- Use ONLY the tools provided
- Use ONLY column names from the list
- Resolve minor misspellings
- Do NOT invent columns or tools
- Return a tool call with valid JSON
"""


def route_user_request(user_message: str, column_types: dict):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": build_system_prompt(column_types)},
            {"role": "user", "content": user_message}
        ],
        tools=CLEANING_TOOLS,
        tool_choice="auto"
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return {
            "tool_name": tc.function.name,
            "arguments": json.loads(tc.function.arguments)
        }

    return None
