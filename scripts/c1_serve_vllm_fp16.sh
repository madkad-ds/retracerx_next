#!/usr/bin/env bash
# c1_serve_vllm_fp16.sh — serve Qwen 3.5-9B on vLLM in FP16 (NOT AWQ), OpenAI-
# compatible on :8000, with xgrammar guided decoding. FP16 keeps outputs
# numerically consistent with the in-process transformers runs (AWQ would shift
# them); AWQ stays the S1 benchmark (c1_serve_reasoner.sh).
#
# ~18 GB weights on a 24 GB L4; paged-attention KV cache fills the rest.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   export REASONER_MODEL_REPO=Qwen/Qwen3.5-9B
#   bash phaseC/stretch/S1_serving/c1_serve_vllm_fp16.sh
#   # then, in the app shell:  export REASONER_BACKEND=vllm
set -euo pipefail
MODEL="${REASONER_MODEL_REPO:-Qwen/Qwen3.5-9B}"
SERVED="${REASONER_SERVED_NAME:-reasoner}"
MAXLEN="${REASONER_MAX_MODEL_LEN:-8192}"
BACKEND="${GUIDED_BACKEND:-xgrammar}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"     # PIN a tag for reproducibility
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
API_KEY="${VLLM_API_KEY:-retracerx-local}"
# STRICT=1 adds --enforce-eager for tighter run-to-run determinism (a bit slower).
EAGER=""; [ "${STRICT:-0}" = "1" ] && EAGER="--enforce-eager"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52}" #pin digest

echo ">> serving ${MODEL} (FP16) as '${SERVED}', guided=${BACKEND} ${EAGER:+(+enforce-eager)}"
docker run -d --rm --name retracerx-vllm \
  --gpus '"device=0"' --ipc=host --shm-size=8g -p 8000:8000 \
  -e HF_TOKEN="${HF_TOKEN:-}" -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_API_KEY="${API_KEY}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  --model "${MODEL}" --served-model-name "${SERVED}" \
  --dtype float16 \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization 0.90 \
  --seed 20260727 --enable-prefix-caching ${EAGER} \
  --port 8000

echo ">> follow load:  docker logs -f retracerx-vllm   (wait for 'Application startup complete')"
echo ">> then:         export REASONER_BACKEND=vllm && python scripts/c1_smoke_reasoner.py"
