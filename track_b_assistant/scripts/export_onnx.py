"""
Converts the MiniLM embedding model to ONNX and makes a quantized (INT8) version.

    python scripts/export_onnx.py

Output (models/minilm-onnx/):
    model.onnx            - FP32 ONNX model
    model_quantized.onnx  - INT8 dynamic quantization (smaller + faster on CPU)
    tokenizer files

Why only the embedding model and not the LLM?
    The LLM runs on Groq (a hosted API) so we don't have the weights, and the local
    LLM is served with vLLM which already has its own optimized GPU kernels
    (PagedAttention, continuous batching). Exporting a multi-billion parameter LLM
    to ONNX would not make it faster than vLLM. The embedding model runs inside
    our own container on CPU for every request, so optimizing it is useful.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

OUTPUT_DIR = config.ONNX_MODEL_DIR


class EncoderWrapper(torch.nn.Module):
    """
    torch.onnx.export passes inputs as positional arguments, but newer transformers
    versions changed the order of BertModel.forward() arguments. This wrapper calls
    the model with keyword arguments and returns only last_hidden_state.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return output.last_hidden_state


def export():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {config.EMBEDDING_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)
    model.eval()

    sample = tokenizer(["This is a sample sentence"], return_tensors="pt")
    onnx_path = OUTPUT_DIR / "model.onnx"

    print("Exporting to ONNX ...")
    with torch.no_grad():
        torch.onnx.export(
            EncoderWrapper(model),
            (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            # batch size and sentence length can change at runtime
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
            dynamo=False,
        )
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("Quantizing to INT8 ...")
    quantized_path = OUTPUT_DIR / "model_quantized.onnx"
    quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)

    size_fp32 = onnx_path.stat().st_size / 1e6
    size_int8 = quantized_path.stat().st_size / 1e6
    print(f"model.onnx: {size_fp32:.1f} MB, model_quantized.onnx: {size_int8:.1f} MB")


def verify():
    """Checks that the ONNX embeddings are almost the same as the original model."""
    from sentence_transformers import SentenceTransformer

    from app.rag.embeddings import Embedder

    sentences = ["How many paid leaves do I get?", "Reset my VPN password", "CloudDrive Business plan price"]
    original = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu").encode(sentences, normalize_embeddings=True)

    config.USE_ONNX_EMBEDDINGS = True
    for file_name in ["model.onnx", "model_quantized.onnx"]:
        config.ONNX_MODEL_FILE = file_name
        onnx_vectors = np.array(Embedder().embed(sentences))
        similarity = (original * onnx_vectors).sum(axis=1)  # vectors are normalized, so dot = cosine
        print(f"{file_name}: cosine similarity with PyTorch model = {similarity.round(4).tolist()}")


if __name__ == "__main__":
    export()
    verify()
