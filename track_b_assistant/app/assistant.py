"""
The main assistant logic:

    1. Retrieve relevant chunks from the vector DB (RAG)
    2. Build the prompt (system prompt + chat history + context + question)
    3. Call the LLM with tools. If the model asks for a tool, run it and send the result back.
    4. Parse and validate the final JSON answer
    5. If the LLM is completely down, return the retrieved documents instead (graceful degradation)
"""
import asyncio
import json
import logging
import re

from pydantic import ValidationError

from app import config
from app.llm_client import AllProvidersFailedError, chat_completion
from app.prompts import JSON_REPAIR_PROMPT, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_context
from app.rag import vector_store
from app.schemas import AssistantAnswer
from app.tools import TOOL_DEFINITIONS, run_tool

logger = logging.getLogger(__name__)


def parse_json_answer(text):
    """
    Tries to turn the model output into an AssistantAnswer.
    Sometimes models wrap JSON in ```json ... ``` or add a sentence before it,
    so we cut out the part between the first "{" and the last "}".
    Returns None if it is not valid.
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
        return AssistantAnswer(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def build_degraded_answer(chunks):
    """Answer used when no LLM is available: show the best matching document passages."""
    if not chunks:
        answer = "Sorry, the AI service is temporarily unavailable. Please try again in a few minutes."
    else:
        passages = "\n\n".join(f"- ({c['source']}) {c['text'][:300]}..." for c in chunks[:2])
        answer = (
            "The AI service is temporarily unavailable, but these document passages "
            "look relevant to your question:\n\n" + passages
        )
    return {
        "answer": answer,
        "sources": sorted({c["source"] for c in chunks}),
        "confidence": "low",
        "follow_up_questions": [],
        "model_used": None,
        "tools_used": [],
        "degraded": True,
        "total_tokens": 0,
    }


def tool_call_to_dict(tool_call):
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
    }


async def answer_question(question, history=None, temperature=None, top_p=None):
    # ---- step 1: retrieval ----
    # embedding is CPU work, so run it in a thread to not block other requests
    try:
        chunks = await asyncio.to_thread(vector_store.search, question, config.TOP_K)
    except Exception as e:
        logger.error("Retrieval failed, continuing without context: %s", e)
        chunks = []

    # ---- step 2: prompt ----
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (history or [])[-6:]:  # only the last few messages to keep the prompt small
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(context=format_context(chunks), question=question),
    })

    tools_used = []
    model_used = None
    final_text = None
    total_tokens = 0  # W16: used by the evaluation harness to compare single-agent vs multi-agent cost

    def _add_tokens(response):
        nonlocal total_tokens
        usage = getattr(response, "usage", None)
        if usage and getattr(usage, "total_tokens", None):
            total_tokens += usage.total_tokens

    try:
        # ---- step 3: LLM + tool calling loop ----
        for _ in range(config.MAX_TOOL_ROUNDS):
            response, model_used = await chat_completion(
                messages, tools=TOOL_DEFINITIONS, temperature=temperature, top_p=top_p
            )
            _add_tokens(response)
            message = response.choices[0].message

            if not message.tool_calls:
                final_text = message.content
                break

            # the model wants to call one or more tools
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tool_call_to_dict(tc) for tc in message.tool_calls],
            })
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                logger.info("Tool call: %s(%s)", name, arguments)
                result = await asyncio.to_thread(run_tool, name, arguments)
                tools_used.append(name)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})

        if final_text is None:
            # too many tool rounds, ask for the final answer without tools
            response, model_used = await chat_completion(
                messages, json_mode=True, temperature=temperature, top_p=top_p
            )
            _add_tokens(response)
            final_text = response.choices[0].message.content

        # ---- step 4: structured output ----
        parsed = parse_json_answer(final_text)
        if parsed is None:
            logger.warning("Model did not return valid JSON, asking it to fix the format")
            repair_messages = [{"role": "user", "content": JSON_REPAIR_PROMPT.format(text=final_text)}]
            response, model_used = await chat_completion(repair_messages, json_mode=True, temperature=0)
            _add_tokens(response)
            parsed = parse_json_answer(response.choices[0].message.content)

        if parsed is None:
            # last option: use the raw text as the answer
            parsed = AssistantAnswer(answer=final_text or "Sorry, I could not generate an answer.", confidence="low")

    except AllProvidersFailedError as e:
        # ---- step 5: graceful degradation ----
        logger.error("LLM unavailable: %s", e)
        return build_degraded_answer(chunks)

    result = parsed.model_dump()
    if not result["sources"]:
        # fall back to the documents we retrieved, but only if the model was confident enough to use them
        if chunks and result["confidence"] != "low":
            result["sources"] = sorted({c["source"] for c in chunks})
    result["model_used"] = model_used
    result["tools_used"] = tools_used
    result["degraded"] = False
    result["total_tokens"] = total_tokens
    return result
