"""
Document ingestion pipeline:
    load files -> split into chunks -> create embeddings -> store in ChromaDB

Run it with:
    python -m app.rag.ingest            (add / update documents)
    python -m app.rag.ingest --reset    (delete everything and ingest again)
"""
import argparse
import logging
import time

from pypdf import PdfReader

from app import config
from app.rag import vector_store

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf"]


def load_documents(data_dir):
    """Reads all supported files from the folder. Returns list of (filename, text)."""
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                reader = PdfReader(str(path))
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Could not read %s: %s", path.name, e)
            continue

        if text.strip():
            documents.append((path.name, text))
    return documents


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Splits text into chunks of about chunk_size characters.

    We first split on blank lines (paragraphs) so that we don't cut sentences
    in the middle when possible. Paragraphs are joined until the chunk is full.
    The last `overlap` characters of a chunk are repeated at the start of the
    next chunk so that context is not lost at the borders.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # keep a heading together with the paragraph below it, otherwise a chunk can end
    # with "## 3. Leave Policy" and the actual leave details go to the next chunk
    paragraphs = []
    pending_heading = ""
    for para in raw_paragraphs:
        if para.startswith("#") and "\n" not in para:
            pending_heading = pending_heading + "\n" + para if pending_heading else para
            continue
        if pending_heading:
            para = pending_heading + "\n" + para
            pending_heading = ""
        paragraphs.append(para)
    if pending_heading:
        paragraphs.append(pending_heading)

    # very long paragraphs are cut into smaller pieces first
    pieces = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            pieces.append(para)
        else:
            step = chunk_size - overlap
            for start in range(0, len(para), step):
                pieces.append(para[start:start + chunk_size])

    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > chunk_size:
            chunks.append(current)
            # start the new chunk with the tail of the previous one
            tail = current[-overlap:] if overlap > 0 else ""
            # try to start the tail at a word boundary
            if " " in tail:
                tail = tail[tail.index(" ") + 1:]
            current = tail + "\n\n" + piece if tail else piece
        else:
            current = current + "\n\n" + piece if current else piece

    if current:
        chunks.append(current)
    return chunks


def ingest_documents(reset=False):
    start = time.time()
    if reset:
        vector_store.reset_collection()

    documents = load_documents(config.DATA_DIR)
    if not documents:
        logger.warning("No documents found in %s", config.DATA_DIR)
        return {"documents": 0, "chunks": 0}

    total_chunks = 0
    for filename, text in documents:
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        ids = [f"{filename}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
        vector_store.add_chunks(ids, chunks, metadatas)
        logger.info("Ingested %s -> %d chunks", filename, len(chunks))
        total_chunks += len(chunks)

    seconds = round(time.time() - start, 2)
    logger.info("Ingestion finished: %d documents, %d chunks in %ss", len(documents), total_chunks, seconds)
    return {"documents": len(documents), "chunks": total_chunks, "seconds": seconds}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database")
    parser.add_argument("--reset", action="store_true", help="delete existing chunks first")
    args = parser.parse_args()
    print(ingest_documents(reset=args.reset))
