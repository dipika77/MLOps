"""
Embedding model wrapper.

We use sentence-transformers/all-MiniLM-L6-v2 from Hugging Face.
It is small (~90MB, 384 dimensions) and runs fast on CPU.

Two backends are supported:
1. sentence-transformers (PyTorch) - default
2. ONNX Runtime - faster on CPU, enable with USE_ONNX_EMBEDDINGS=true
   (you need to run scripts/export_onnx.py first)
"""
import logging

import numpy as np

from app import config

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self.backend = "pytorch"
        onnx_path = config.ONNX_MODEL_DIR / config.ONNX_MODEL_FILE

        if config.USE_ONNX_EMBEDDINGS and onnx_path.exists():
            import onnxruntime as ort
            from transformers import AutoTokenizer

            logger.info("Loading ONNX embedding model from %s", onnx_path)
            self.tokenizer = AutoTokenizer.from_pretrained(str(config.ONNX_MODEL_DIR))
            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.backend = "onnx"
        else:
            if config.USE_ONNX_EMBEDDINGS:
                logger.warning("ONNX model not found at %s, using PyTorch model instead", onnx_path)
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", config.EMBEDDING_MODEL)
            self.model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    def embed(self, texts):
        """Takes a list of strings and returns a list of vectors (list of floats)."""
        if len(texts) == 0:
            return []
        if self.backend == "onnx":
            vectors = self._embed_onnx(texts)
        else:
            vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def _embed_onnx(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="np")
        inputs = {name: tokens[name].astype(np.int64) for name in self.input_names}
        last_hidden_state = self.session.run(None, inputs)[0]

        # mean pooling (same thing sentence-transformers does for MiniLM)
        mask = tokens["attention_mask"][..., None].astype(np.float32)
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        vectors = summed / counts

        # L2 normalize so cosine similarity works properly
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-9, None)


# load the model only once (it is slow to load)
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
