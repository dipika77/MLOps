"""
Small retrieval evaluation: for each question, check if a chunk containing the
expected answer is in the top-k results. I used this to choose CHUNK_SIZE / CHUNK_OVERLAP.

    python scripts/eval_retrieval.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402
from app.rag.embeddings import get_embedder  # noqa: E402
from app.rag.ingest import chunk_text, load_documents  # noqa: E402

# (question, a phrase that should be inside one of the retrieved chunks)
TEST_QUESTIONS = [
    ("How many paid leaves do I get per year?", "24 days"),
    ("Can I work from home?", "3 days per week"),
    ("How much does the Business plan cost?", "$15"),
    ("My account got locked", "locked for 15 minutes"),
    ("What is the meal allowance when travelling?", "$50 per day"),
    ("How long is maternity leave?", "26 weeks"),
    ("Is there a free trial?", "14-day"),
    ("How do I get VPN?", "SecureConnect"),
    ("Where is my data stored?", "Frankfurt"),
    ("How fast will IT reply to a high priority ticket?", "within 4 hours"),
    ("Can I get a refund?", "pro-rata"),
    ("What is the learning budget?", "$1,000"),
    ("How long does software approval take?", "2 business days"),
    ("When do salary revisions happen?", "January"),
    ("How long are deleted files kept?", "180 days"),
]

SETTINGS_TO_TRY = [(300, 50), (400, 80), (500, 100), (800, 150)]  # (chunk_size, overlap)


def count_hits(chunks, chunk_vectors, question_vectors, k):
    hits = 0
    for i, (question, expected) in enumerate(TEST_QUESTIONS):
        similarities = chunk_vectors @ question_vectors[i]  # vectors are normalized -> cosine similarity
        top_k_indexes = np.argsort(-similarities)[:k]
        if any(expected in chunks[j] for j in top_k_indexes):
            hits += 1
    return hits


def main():
    embedder = get_embedder()
    print(f"Embedding backend: {embedder.backend}")

    documents = load_documents(config.DATA_DIR)
    question_vectors = np.array(embedder.embed([q for q, _ in TEST_QUESTIONS]))

    for chunk_size, overlap in SETTINGS_TO_TRY:
        chunks = []
        for _, text in documents:
            chunks.extend(chunk_text(text, chunk_size, overlap))
        chunk_vectors = np.array(embedder.embed(chunks))

        hit2 = count_hits(chunks, chunk_vectors, question_vectors, k=2)
        hit4 = count_hits(chunks, chunk_vectors, question_vectors, k=4)
        total = len(TEST_QUESTIONS)
        print(f"size={chunk_size} overlap={overlap} chunks={len(chunks)} hit@2={hit2}/{total} hit@4={hit4}/{total}")


if __name__ == "__main__":
    main()
