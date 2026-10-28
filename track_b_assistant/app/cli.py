"""
Simple terminal chat (Task 1 demo, no web server needed).

    python -m app.cli              plain RAG + tool-calling assistant (W15)
    python -m app.cli --agentic    Researcher/Verifier cross-source verification loop (W16)
"""
import asyncio
import json
import logging
import sys

from app.agentic import agentic_answer_question
from app.assistant import answer_question
from app.rag import vector_store
from app.rag.ingest import ingest_documents


async def main():
    logging.basicConfig(level=logging.WARNING)
    agentic_mode = "--agentic" in sys.argv

    if vector_store.count_chunks() == 0:
        print("Vector DB is empty, ingesting documents first...")
        print(ingest_documents())

    print(f"Acme Assistant [{'agentic' if agentic_mode else 'standard'} mode] (type 'exit' to quit)\n")
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        if not question:
            continue

        if agentic_mode:
            result = await agentic_answer_question(question)
        else:
            result = await answer_question(question, history=history)
        print("\nAssistant (JSON):")
        print(json.dumps(result, indent=2))
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    asyncio.run(main())
