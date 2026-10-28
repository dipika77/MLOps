#!/bin/bash
# Serve an open-source LLM locally with vLLM (OpenAI-compatible API on port 8001).
# Needs a machine with an NVIDIA GPU (vLLM does not support Mac GPUs).
#
# After it starts, set this in .env so the assistant uses it as a fallback:
#   VLLM_BASE_URL=http://localhost:8001/v1
#   VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct

MODEL=${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}

# Option 1: with Docker (recommended)
docker run --gpus all --rm -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model "$MODEL" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# Option 2: without Docker
#   pip install vllm
#   vllm serve "$MODEL" --port 8001 --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes
#
# Test it:
#   curl http://localhost:8001/v1/models
