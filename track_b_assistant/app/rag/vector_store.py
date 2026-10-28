"""
Vector database code. We use ChromaDB because it runs locally,
saves data to disk and does not need a separate server.
"""
import logging

import chromadb

from app import config
from app.rag.embeddings import get_embedder

logger = logging.getLogger(__name__)

_client = None


def get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # cosine distance works best with normalized sentence embeddings
    return _client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    """Deletes all stored chunks (used when re-ingesting documents)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        _client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet


def add_chunks(ids, texts, metadatas):
    collection = get_collection()
    embeddings = get_embedder().embed(texts)
    # upsert = insert or update, so running ingestion twice does not create duplicates
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)


def count_chunks():
    return get_collection().count()


def search(query, top_k=None):
    """Returns the top_k most similar chunks for the query."""
    if top_k is None:
        top_k = config.TOP_K

    collection = get_collection()
    if collection.count() == 0:
        logger.warning("Vector store is empty. Did you run the ingestion?")
        return []

    query_vector = get_embedder().embed([query])[0]
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    chunks = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "score": round(1 - distance, 3),  # convert distance to similarity
        })
    return chunks
