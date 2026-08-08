#!/usr/bin/env bash
# serve_env.sh — Phase E session spine (the E-stage analogue of neo4j_env.sh).
#
# Brings up + verifies the demo stack (Neo4j + vLLM) and exports the vars the
# FastAPI/Streamlit app reads, so there is ONE source of truth for the served
# model name and API key. Idempotent: safe to `source` repeatedly (e.g. once per
# tmux pane). SOURCE it — don't execute — so the exports land in your shell:
#     source scripts/serve_env.sh
#
# NOTE: no `set -e/-u` here on purpose — this file is sourced into the interactive
# shell, and a failing check must not kill the user's shell.

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# 1. Neo4j: start + wait-for-Bolt + export NEO4J_* + ping APOC/variant (Phase B).
#    neo4j_env.sh is idempotent; a no-op start if the container is already up.
source "$_here/neo4j_env.sh"

# 2. vLLM config. Track the serve script (so this follows any change there),
#    then fall back to the observed defaults.
_serve="$_here/c1_serve_vllm_fp16.sh"
export REASONER_BACKEND="vllm"                       # the demo REQUIRES the vllm backend (Phase C E-C9)
export VLLM_BASE="http://localhost:8000/v1"

VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$(grep -oE -- '--served-model-name[ =]+[^ ]+' "$_serve" 2>/dev/null | awk '{print $NF}' | head -n1)}"
export VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-reasoner}"

VLLM_API_KEY="${VLLM_API_KEY:-$(grep -oE 'VLLM_API_KEY:-[^}]+' "$_serve" 2>/dev/null | head -n1 | sed 's/.*:-//')}"
export VLLM_API_KEY="${VLLM_API_KEY:-retracerx-local}"

# 3. vLLM: start if down, then wait for the OpenAI endpoint (first boot loads
#    ~18 GB of FP16 weights, so allow a couple of minutes).
if curl -sf -H "Authorization: Bearer $VLLM_API_KEY" "$VLLM_BASE/models" >/dev/null 2>&1; then
  echo "[serve_env] vLLM already up"
else
  echo "[serve_env] vLLM not responding — starting retracerx-vllm…"
  docker start retracerx-vllm >/dev/null 2>&1 \
    || echo "[serve_env] retracerx-vllm missing; in another shell run:  bash scripts/c1_serve_vllm_fp16.sh"
  printf "[serve_env] waiting for vLLM"
  for i in $(seq 1 60); do
    if curl -sf -H "Authorization: Bearer $VLLM_API_KEY" "$VLLM_BASE/models" >/dev/null 2>&1; then
      echo " ready"; break
    fi
    printf "."; sleep 2
    [ "$i" -eq 60 ] && echo " TIMEOUT (still loading? check: docker logs --tail 40 retracerx-vllm)"
  done
fi

echo "[serve_env] ready — NEO4J_URI=${NEO4J_URI:-bolt://localhost:7687}  VLLM_BASE=$VLLM_BASE  model=$VLLM_SERVED_MODEL_NAME  backend=$REASONER_BACKEND"
