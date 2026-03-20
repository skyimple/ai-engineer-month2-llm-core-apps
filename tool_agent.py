"""
Terminal Tool-Calling Agent using DeepSeek with streaming support.
Provides 3 tools: weather, calculator, and notes search.
Features: Multi-turn chat, cost tracking, auto-retry, special commands.
"""

import ast
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import instructor
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models import Invoice

load_dotenv("config.env")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
instructor_client = instructor.from_openai(client)


# =============================================================================
# DeepSeek Pricing & Cost Tracker
# =============================================================================

DEEPSEEK_PRICING = {
    "input": 2.0,   # RMB per million tokens
    "output": 3.0,  # RMB per million tokens
}


class CostTracker:
    """Tracks token usage and calculates accumulated cost in RMB."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

    def update(self, usage: dict):
        """Update totals from an API response usage dict."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        # Calculate cost in RMB
        prompt_cost = (prompt_tokens / 1_000_000) * DEEPSEEK_PRICING["input"]
        completion_cost = (completion_tokens / 1_000_000) * DEEPSEEK_PRICING["output"]
        self.total_cost = (self.total_prompt_tokens / 1_000_000) * DEEPSEEK_PRICING["input"] + \
                          (self.total_completion_tokens / 1_000_000) * DEEPSEEK_PRICING["output"]

    def display(self) -> str:
        """Return a formatted string showing usage and cost."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        return (
            f"\n{'=' * 40}\n"
            f"Token Usage Summary\n"
            f"{'=' * 40}\n"
            f"Prompt tokens:      {self.total_prompt_tokens:,}\n"
            f"Completion tokens:  {self.total_completion_tokens:,}\n"
            f"Total tokens:       {total_tokens:,}\n"
            f"Estimated cost:     {self.total_cost:.4f} RMB\n"
            f"{'=' * 40}\n"
        )


# =============================================================================
# Conversation Memory
# =============================================================================

@dataclass
class ConversationMemory:
    """Stores full conversation history with token/cost tracking."""
    system_prompt: str
    messages: list[dict] = field(default_factory=list)
    cost_tracker: CostTracker = field(default_factory=CostTracker)

    def __post_init__(self):
        # Initialize with system prompt
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def add_user(self, content: str):
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | None = None, tool_calls: list | None = None):
        """Add an assistant message."""
        msg = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str):
        """Add a tool result message."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

    def update_usage(self, usage: dict):
        """Update cost tracker with API usage."""
        self.cost_tracker.update(usage)

    def display_history(self) -> str:
        """Return formatted conversation history with message numbers."""
        lines = []
        lines.append("\n" + "=" * 50)
        lines.append("Conversation History")
        lines.append("=" * 50)

        for i, msg in enumerate(self.messages):
            role = msg["role"]
            if role == "system":
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                lines.append(f"[{i}] SYSTEM: {content}")
            elif role == "user":
                lines.append(f"[{i}] USER: {msg['content']}")
            elif role == "assistant":
                if msg.get("tool_calls"):
                    tc_names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                    lines.append(f"[{i}] ASSISTANT (tools: {', '.join(tc_names)})")
                else:
                    content = msg.get("content", "")
                    content = content[:100] + "..." if len(content) > 100 else content
                    lines.append(f"[{i}] ASSISTANT: {content}")
            elif role == "tool":
                content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                lines.append(f"[{i}] TOOL (id={msg['tool_call_id']}): {content}")
            lines.append("-" * 40)

        lines.append(f"\nTotal messages: {len(self.messages)}")
        return "\n".join(lines)


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

def process_streaming_response(stream) -> tuple[str, list | None, dict | None]:
    """
    Process a streaming response from DeepSeek.
    Returns (full_content, tool_calls_to_execute, usage).
    tool_calls_to_execute is None if no tool calls, otherwise list of {id, name, arguments}.
    usage is the usage dict from the API response (only available on final chunk).
    """
    content_accumulator = ""
    tool_calls_accumulator = {}  # index -> {id, name, arguments}
    usage = None

    for chunk in stream:
        # Extract usage from the last chunk if available
        if hasattr(chunk, 'usage') and chunk.usage:
            usage = chunk.usage

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

    return content_accumulator, tool_calls_to_execute, usage


# =============================================================================
# Retry-enabled API call
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, Exception)),
    reraise=True
)
def create_chat_completion(messages: list, tools: list, tool_choice: str = "auto"):
    """
    Create a chat completion with automatic retry on transient errors.
    Retries on: 429 (rate limit), 500 (server error), timeout.
    """
    try:
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True
        )
        return stream
    except Exception as e:
        error_str = str(e).lower()
        # Retry on rate limit (429) or server error (500)
        if "429" in error_str or "500" in error_str or "timeout" in error_str or "rate" in error_str:
            raise
        # Re-raise on other errors
        raise


# =============================================================================
# Main Agent Loop
# =============================================================================

def run_agent():
    """Run the interactive terminal agent loop."""

    system_prompt = """You are a helpful assistant with access to tools. When a user asks a question:

1. If they ask about weather, use the get_weather tool with the city name.
2. If they ask to calculate or compute something, use the calculator tool.
3. If they ask to search notes or find something in notes, use the search_notes tool.
4. If they want to parse an invoice, carefully extract the details from the invoice text they provide.

Be concise and helpful in your responses."""

    memory = ConversationMemory(system_prompt=system_prompt)

    print("=" * 60)
    print("Terminal Tool-Calling Agent (DeepSeek)")
    print("Tools: get_weather, calculator, search_notes")
    print("Commands: 'cost' - show usage | 'history' - show conversation")
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

        # Handle special commands
        if user_input.lower() == "cost":
            print(memory.cost_tracker.display())
            continue

        if user_input.lower() == "history":
            print(memory.display_history())
            print()
            continue

        # Add user message to memory
        memory.add_user(user_input)

        # Send to DeepSeek with streaming and retry
        try:
            stream = create_chat_completion(
                messages=memory.messages,
                tools=TOOLS,
                tool_choice="auto"
            )

            full_content, tool_calls, usage = process_streaming_response(stream)

            # Update cost tracking
            if usage:
                memory.update_usage({
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens
                })
                total_tokens = usage.prompt_tokens + usage.completion_tokens
                print(f"\n[Tokens used: {total_tokens}]")

            # Build assistant message with content and/or tool_calls
            # IMPORTANT: when tool_calls exist, content should be None (not empty string)
            if tool_calls:
                # Assistant message with tool calls only (no content)
                assistant_tool_calls = [
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
                memory.messages.append({
                    "role": "assistant",
                    "tool_calls": assistant_tool_calls
                })
            elif full_content:
                # Assistant message with content only (no tool calls)
                memory.messages.append({
                    "role": "assistant",
                    "content": full_content
                })

            # Handle tool calls
            if tool_calls:
                for tc in tool_calls:
                    print(f"\n[Calling tool: {tc['name']} with args: {tc['arguments']}]")
                    result = execute_tool(tc["name"], tc["arguments"])
                    print(f"[Tool result]: {result}\n")

                    # Append tool result as a special message
                    memory.add_tool_result(tc["id"], result)

                # Continue: get the model's response to tool results
                print("[Model is thinking...]\n")
                stream = create_chat_completion(
                    messages=memory.messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )
                full_content, _, usage = process_streaming_response(stream)

                if usage:
                    memory.update_usage({
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens
                    })
                    total_tokens = usage.prompt_tokens + usage.completion_tokens
                    print(f"\n[Tokens used: {total_tokens}]")

                if full_content:
                    memory.add_assistant(full_content)

        except Exception as e:
            print(f"Error: {str(e)}")
            # Remove the failed user message from memory
            if memory.messages and memory.messages[-1]["role"] == "user":
                memory.messages.pop()
            continue

        print()


if __name__ == "__main__":
    run_agent()
