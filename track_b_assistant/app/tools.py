"""
Tools (functions) the LLM can call.

TOOL_DEFINITIONS is the JSON schema we send to the LLM (OpenAI function calling format,
Groq and vLLM both support it). run_tool() runs the function the model asked for.
"""
import ast
import json
import logging
import operator
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from app import config
from app.rag import vector_store

logger = logging.getLogger(__name__)


# ---------------- tool 1: search documents ----------------
def search_documents(query):
    chunks = vector_store.search(query, top_k=config.TOP_K)
    if not chunks:
        return "No matching documents found."
    return "\n\n".join(f"(source: {c['source']}) {c['text']}" for c in chunks)


# ---------------- tool 2: calculator ----------------
# we don't use eval() because it can run any python code, so we only allow maths operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        if isinstance(node.op, ast.Pow) and abs(_evaluate(node.right)) > 100:
            raise ValueError("exponent too large")
        return ALLOWED_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression):
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
        if isinstance(result, float):
            result = round(result, 4)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: could not calculate '{expression}' ({e})"


# ---------------- tool 3: current date and time ----------------
def get_current_datetime(timezone="UTC"):
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        return f"Error: unknown timezone '{timezone}'. Use a name like 'Asia/Kolkata' or 'UTC'."
    return now.strftime("%A, %d %B %Y, %H:%M (%Z)")


# ---------------- tool 4: weather (external API) ----------------
def get_weather(city):
    try:
        # wttr.in is a free weather service, no API key needed
        response = requests.get(f"https://wttr.in/{city}", params={"format": "3"}, timeout=5)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        logger.warning("Weather API failed: %s", e)
        return "Error: weather service is not reachable right now."


# ---------------- tool 5: create IT support ticket ----------------
def create_support_ticket(title, description, priority="medium"):
    if priority not in ["low", "medium", "high"]:
        priority = "medium"

    ticket = {
        "ticket_id": "TICKET-" + uuid.uuid4().hex[:6].upper(),
        "title": title,
        "description": description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    tickets = []
    if config.TICKETS_FILE.exists():
        try:
            tickets = json.loads(config.TICKETS_FILE.read_text())
        except json.JSONDecodeError:
            tickets = []
    tickets.append(ticket)
    config.TICKETS_FILE.write_text(json.dumps(tickets, indent=2))

    return f"Ticket {ticket['ticket_id']} created with priority '{priority}'."


# ---------------- tool schemas for the LLM ----------------
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search Acme Corp company documents (HR policy, IT support guide, CloudDrive product FAQ).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a maths expression, for example '24 - 7' or '15 * 12 * 0.8'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Maths expression using + - * / ** %"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time in a timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone like 'Asia/Kolkata' or 'UTC'"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Mumbai'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create an IT support ticket when the user asks to raise/report an IT issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title of the issue"},
                    "description": {"type": "string", "description": "Details of the issue"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["title", "description"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "calculator": calculator,
    "get_current_datetime": get_current_datetime,
    "get_weather": get_weather,
    "create_support_ticket": create_support_ticket,
}


def run_tool(name, arguments):
    """Runs a tool by name. Errors are returned as text so the LLM can see them."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"Error: tool '{name}' does not exist."
    try:
        return function(**arguments)
    except TypeError as e:
        return f"Error: wrong arguments for {name}: {e}"
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return f"Error: tool {name} failed: {e}"
