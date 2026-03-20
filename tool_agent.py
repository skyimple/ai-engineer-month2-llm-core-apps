"""
Terminal Tool-Calling Agent using DeepSeek with streaming support.
Provides 3 tools: weather, calculator, and notes search.
"""

import ast
import json
import os
import re
import sys
import urllib.request
from typing import Any

import instructor
from openai import OpenAI
from dotenv import load_dotenv

from models import Invoice

load_dotenv("config.env")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
instructor_client = instructor.from_openai(client)


# =============================================================================
# Tool Definitions (OpenAI-compatible JSON Schema format)
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetches the current weather for a specified city using Open-Meteo API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a mathematical expression safely using AST parsing. Supports: +, -, *, /, **, %.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression to evaluate, e.g., '25 * 4 + 10'."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Searches for a query string within all .txt files in the notes/ directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The text to search for in note files."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# =============================================================================
# Tool Implementations
# =============================================================================

def get_weather(city: str) -> str:
    """Fetch weather for a city using Open-Meteo API (no API key required)."""
    try:
        # Step 1: Geocode the city
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        with urllib.request.urlopen(geocode_url, timeout=10) as response:
            geocode_data = json.loads(response.read().decode())

        if not geocode_data.get("results"):
            return f"City '{city}' not found."

        result = geocode_data["results"][0]
        lat, lon = result["latitude"], result["longitude"]
        city_name = result.get("name", city)

        # Step 2: Get current weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        with urllib.request.urlopen(weather_url, timeout=10) as response:
            weather_data = json.loads(response.read().decode())

        cw = weather_data.get("current_weather", {})
        temp = cw.get("temperature", "N/A")
        wind_speed = cw.get("windspeed", "N/A")
        weather_code = cw.get("weathercode", 0)

        # Simple weather code mapping
        weather_desc = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
        }.get(weather_code, f"Code {weather_code}")

        return f"Weather in {city_name}: {weather_desc}, Temperature: {temp}°C, Wind: {wind_speed} km/h"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


def calculator(expression: str) -> str:
    """Safely evaluate a math expression using AST parsing."""
    try:
        # Parse the expression
        tree = ast.parse(expression, mode="eval")

        # Whitelist of allowed AST nodes
        allowed_nodes = {
            ast.Expression,
            ast.Constant,  # Python 3.8+
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Pow,
            ast.Mod,
            ast.USub,
            ast.UAdd,
        }

        def check_node(node: ast.AST) -> bool:
            """Recursively check that all nodes are allowed."""
            if type(node) not in allowed_nodes:
                return False
            if isinstance(node, ast.Constant):
                # Only allow numeric constants
                return isinstance(node.value, (int, float))
            if isinstance(node, ast.BinOp):
                return check_node(node.left) and check_node(node.right)
            if isinstance(node, ast.UnaryOp):
                return check_node(node.operand)
            return True

        if not check_node(tree.body):
            return "Error: Expression contains disallowed operations."

        # Evaluate the expression
        result = eval(compile(tree, filename="", mode="eval"))

        # Format result
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 10)
        return str(result)
    except SyntaxError:
        return "Error: Invalid expression syntax."
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: {str(e)}"


def search_notes(query: str) -> str:
    """Search for query in all .txt files within the notes/ directory."""
    notes_dir = os.path.join(os.path.dirname(__file__), "notes")

    if not os.path.isdir(notes_dir):
        return "Error: notes/ directory not found."

    results = []
    txt_files = [f for f in os.listdir(notes_dir) if f.endswith(".txt")]

    if not txt_files:
        return "No .txt files found in notes/ directory."

    for filename in txt_files:
        filepath = os.path.join(notes_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    if query.lower() in line.lower():
                        results.append(f"[{filename}:{lineno}] {line.rstrip()}")
        except Exception as e:
            results.append(f"[{filename}] Error reading file: {str(e)}")

    if not results:
        return f"No matches found for '{query}' in notes/."

    return "\n".join(results)


# =============================================================================
# Tool Executor
# =============================================================================

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool by name with given arguments."""
    if tool_name == "get_weather":
        return get_weather(tool_args.get("city", ""))
    elif tool_name == "calculator":
        return calculator(tool_args.get("expression", ""))
    elif tool_name == "search_notes":
        return search_notes(tool_args.get("query", ""))
    else:
        return f"Error: Unknown tool '{tool_name}'"


# =============================================================================
# Streaming Handler
# =============================================================================

def process_streaming_response(stream) -> tuple[str, list | None]:
    """
    Process a streaming response from DeepSeek.
    Returns (full_content, tool_calls_to_execute).
    tool_calls_to_execute is None if no tool calls, otherwise list of {id, name, arguments}.
    """
    content_accumulator = ""
    tool_calls_accumulator = {}  # index -> {id, name, arguments}

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Accumulate content
        if delta.content:
            content_accumulator += delta.content
            # Stream to stdout in real-time with encoding error handling
            try:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
            except UnicodeEncodeError:
                # Fallback: encode with error handling for special chars
                sys.stdout.write(delta.content.encode('utf-8', errors='replace').decode('utf-8'))
                sys.stdout.flush()

        # Collect tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_accumulator:
                    tool_calls_accumulator[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls_accumulator[idx]["id"] = tc.id
                if tc.function and tc.function.name:
                    tool_calls_accumulator[idx]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    tool_calls_accumulator[idx]["arguments"] += tc.function.arguments

    print()  # Newline after streaming content

    # Parse tool call arguments
    tool_calls_to_execute = None
    if tool_calls_accumulator:
        tool_calls_to_execute = []
        for idx in sorted(tool_calls_accumulator.keys()):
            tc = tool_calls_accumulator[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": tc["arguments"]}
            tool_calls_to_execute.append({
                "id": tc["id"],
                "name": tc["name"],
                "arguments": args
            })

    return content_accumulator, tool_calls_to_execute


# =============================================================================
# Main Agent Loop
# =============================================================================

def run_agent():
    """Run the interactive terminal agent loop."""
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant with access to tools. When a user asks a question:

1. If they ask about weather, use the get_weather tool with the city name.
2. If they ask to calculate or compute something, use the calculator tool.
3. If they ask to search notes or find something in notes, use the search_notes tool.
4. If they want to parse an invoice, carefully extract the details from the invoice text they provide.

Be concise and helpful in your responses."""
        }
    ]

    print("=" * 60)
    print("Terminal Tool-Calling Agent (DeepSeek)")
    print("Tools: get_weather, calculator, search_notes")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Append user message
        messages.append({"role": "user", "content": user_input})

        # Send to DeepSeek with streaming
        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True
            )

            full_content, tool_calls = process_streaming_response(stream)

            # Build assistant message with content and/or tool_calls
            if full_content or tool_calls:
                assistant_msg = {"role": "assistant"}
                if full_content:
                    assistant_msg["content"] = full_content
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"])
                            }
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

            # Handle tool calls
            if tool_calls:
                for tc in tool_calls:
                    print(f"\n[Calling tool: {tc['name']} with args: {tc['arguments']}]")
                    result = execute_tool(tc["name"], tc["arguments"])
                    print(f"[Tool result]: {result}\n")

                    # Append tool result as a special message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result
                    })

                # Continue: get the model's response to tool results
                print("[Model is thinking...]\n")
                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=True
                )
                full_content, _ = process_streaming_response(stream)
                if full_content:
                    messages.append({"role": "assistant", "content": full_content})

        except Exception as e:
            print(f"Error: {str(e)}")
            messages.pop()  # Remove the failed user message
            continue

        print()


if __name__ == "__main__":
    run_agent()
