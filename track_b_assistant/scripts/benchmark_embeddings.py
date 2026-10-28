"""
Compares embedding speed: PyTorch vs ONNX FP32 vs ONNX INT8.

    python scripts/benchmark_embeddings.py
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402
from app.rag.embeddings import Embedder  # noqa: E402

QUERIES = [
    "How many paid leaves do I get per year?",
    "What is the refund policy for annual plans?",
    "My laptop is not booting, what should I do?",
    "Can I work from home during probation?",
]
RUNS = 50


def benchmark(name, embedder):
    embedder.embed(QUERIES)  # warm up
    single_times = []
    for i in range(RUNS):
        start = time.perf_counter()
        embedder.embed([QUERIES[i % len(QUERIES)]])
        single_times.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    for _ in range(10):
        embedder.embed(QUERIES * 8)  # batch of 32
    batch_ms = (time.perf_counter() - start) * 1000 / 10

    print(f"{name:<22} single query p50 = {statistics.median(single_times):6.2f} ms | "
          f"batch of 32 = {batch_ms:7.2f} ms")


if __name__ == "__main__":
    config.USE_ONNX_EMBEDDINGS = False
    benchmark("PyTorch", Embedder())

    config.USE_ONNX_EMBEDDINGS = True
    for file_name in ["model.onnx", "model_quantized.onnx"]:
        config.ONNX_MODEL_FILE = file_name
        embedder = Embedder()
        if embedder.backend != "onnx":
            print("ONNX model not found, run scripts/export_onnx.py first")
            break
        benchmark(f"ONNX ({file_name})", embedder)
