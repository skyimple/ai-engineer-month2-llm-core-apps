import json
import re
from datetime import date

from models import Invoice, Item
from client import instructor_client


SYSTEM_PROMPT = """You are an expert invoice parsing system. Extract structured data from invoices.
Return ONLY valid JSON that conforms to the schema. No explanations, no markdown, no additional text."""


def parse_with_instructor(invoice_text: str) -> Invoice:
    """Parse using Instructor for structured output - returns Pydantic object directly."""
    response = instructor_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this invoice:\n{invoice_text}"}
        ],
        response_model=Invoice,
        temperature=0.1,
    )
    return response


def parse_with_normal_prompt(invoice_text: str) -> Invoice:
    """Parse using normal prompt, then manually parse JSON response."""
    from openai import OpenAI
    from dotenv import load_dotenv
    import os

    load_dotenv("config.env")
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this invoice:\n{invoice_text}"}
        ],
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content.strip()

    # Try to extract JSON from the response
    json_match = re.search(r'\{[\s\S]*\}', raw_text)
    if json_match:
        json_str = json_match.group()
    else:
        json_str = raw_text

    data = json.loads(json_str)

    # Parse due_date if present
    if "due_date" in data and data["due_date"]:
        data["due_date"] = date.fromisoformat(data["due_date"])

    # Parse items
    parsed_items = []
    for item_data in data.get("items", []):
        parsed_items.append(Item(**item_data))
    data["items"] = parsed_items

    return Invoice(**data)
