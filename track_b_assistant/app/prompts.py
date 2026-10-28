"""
Prompt templates.

Prompt engineering notes (things I tried while building this):
- Telling the model to ONLY use the context reduced made-up answers a lot.
- Giving the exact JSON format with an example made the output parse correctly
  almost every time. We still validate it with pydantic in assistant.py.
- temperature=0.2 and top_p=0.9 gave consistent, factual answers.
  Higher temperature (0.8+) started adding details that were not in the docs.
"""

SYSTEM_PROMPT = """You are "Acme Assistant", a helpful AI assistant for employees and customers of Acme Corp.

Your job:
- Answer questions about Acme Corp HR policies, IT support and the Acme CloudDrive product.
- Use the CONTEXT given in the user message. The context comes from company documents.
- If the context does not contain the answer, you may call the search_documents tool with a better query.
- If the answer is still not in the documents, say you don't know. Never make up policies, prices or numbers.
- Use the other tools when they help (calculator for maths, get_current_datetime for dates,
  get_weather for weather, create_support_ticket when the user asks to raise an IT ticket).
- Keep answers short and clear (2-5 sentences, or a short bullet list).

OUTPUT FORMAT:
When you give the final answer, reply with ONLY a valid JSON object, no markdown and no extra text:
{
  "answer": "<your answer for the user>",
  "sources": ["<file names of documents you used, empty list if none>"],
  "confidence": "<high | medium | low>",
  "follow_up_questions": ["<1 to 3 short related questions the user might ask next>"]
}
"""

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}"""

JSON_REPAIR_PROMPT = """Convert the following text into a valid JSON object with exactly these keys:
"answer" (string), "sources" (list of strings), "confidence" ("high", "medium" or "low"),
"follow_up_questions" (list of strings). Return only the JSON.

TEXT:
{text}"""


def format_context(chunks):
    if not chunks:
        return "No relevant documents were found."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(parts)
