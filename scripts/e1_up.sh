#!/usr/bin/env bash
# e1_up.sh — bring up the deployed stack: the TWO containers (Neo4j + vLLM).
#
# The host app (FastAPI/Streamlit) is launched separately (E3) from the serve
# venv; this script only stands up the container spine and gates on readiness.
#
# Run from the repo root:  bash scripts/e1_up.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
COMPOSE="$REPO/scripts/docker-compose.phaseE.yml"     # flat layout
SERVE_SH="$REPO/scripts/c1_serve_vllm_fp16.sh"        # canonical FP16 serve command / pinned digest

# Resolve the DIGEST-pinned image: prefer a vllm.env if present, else grep the serve script.
VLLM_IMAGE=""
for f in "$REPO/scripts/vllm.env" "$REPO/vllm.env" "$SERVE_SH"; do
  [[ -f "$f" ]] || continue
  VLLM_IMAGE="$(grep -oE 'vllm/vllm-openai@sha256:[0-9a-f]{64}' "$f" | head -n1 || true)"
  [[ -n "$VLLM_IMAGE" ]] && { echo "==> VLLM_IMAGE from $f"; break; }
done
[[ -n "$VLLM_IMAGE" ]] || { echo "ERROR: no digest-pinned vllm image found in vllm.env or $SERVE_SH"; exit 1; }
export VLLM_IMAGE

# --- 1. Neo4j bind ownership (Phase A/B footgun) -----------------------------
# The neo4j entrypoint reclaims plugins/import as uid 7474 on boot. Before we
# touch them (e.g. swap an APOC jar) reclaim them for the current user.
# NEVER chown neo4j/data — the database must own it.
echo "==> Reclaiming neo4j/plugins and neo4j/import (NOT data)"
sudo chown -R "$(id -u):$(id -g)" neo4j/plugins neo4j/import 2>/dev/null || \
  echo "   (skipped chown — no sudo or already owned; fine if you're not swapping jars)"

# --- 2. Bring up the two containers (digest-pinned vLLM) ----------------------
echo "==> docker compose up -d neo4j vllm  (VLLM_IMAGE=$VLLM_IMAGE)"
docker compose -f "$COMPOSE" up -d neo4j vllm

# --- 3. Wait for Neo4j Bolt + assert APOC and the FILTERED variant -----------
echo "==> Waiting for Neo4j Bolt on :7687"
for i in $(seq 1 60); do
  if docker exec retracerx-neo4j cypher-shell -u neo4j -p retracerx 'RETURN 1' >/dev/null 2>&1; then
    echo "   Bolt ready"; break
  fi
  sleep 2
  [[ $i -eq 60 ]] && { echo "ERROR: Neo4j did not become ready"; exit 1; }
done

echo "==> Pinging APOC + asserting the leakage-filtered variant"
docker exec retracerx-neo4j cypher-shell -u neo4j -p retracerx \
  "RETURN apoc.version() AS apoc" || { echo "ERROR: APOC not loaded"; exit 1; }
# Filtered variant gate: a DRG_DIS__ leakage composite MUST return zero (Phase A A3).
LEAK=$(docker exec retracerx-neo4j cypher-shell -u neo4j -p retracerx --format plain \
  "MATCH ()-[r:DRG_DIS__INDICATION]-() RETURN count(r) AS n" | tail -n1 | tr -dc '0-9')
if [[ "${LEAK:-0}" != "0" ]]; then
  echo "ERROR: leakage type present (count=$LEAK) — this is the FULL graph, not filtered."; exit 1
fi
echo "   filtered variant confirmed (leakage count = 0)"

# --- 4. Wait for vLLM OpenAI endpoint ----------------------------------------
echo "==> Waiting for vLLM on :8000 (first boot loads ~18 GB of FP16 weights)"
for i in $(seq 1 120); do
  if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "   vLLM ready"; break
  fi
  sleep 5
  [[ $i -eq 120 ]] && { echo "ERROR: vLLM did not become ready"; docker logs --tail 40 retracerx-vllm; exit 1; }
done

# --- 5. Deep healthcheck (JSON round-trip, determinism, GPU budget) ----------
echo "==> Running e1_healthcheck.py"
if [[ -x "$REPO/.venv-serve/bin/python" ]]; then
  PY="$REPO/.venv-serve/bin/python"
else
  PY="python3"
fi
"$PY" "$REPO/scripts/e1_healthcheck.py"

echo "✅ E1 stack up: Neo4j (filtered, APOC) + vLLM (Qwen3.5-9B FP16)."
echo "   Next (E3): launch the host app —"
echo "     source .venv-serve/bin/activate && source scripts/neo4j_env.sh"
echo "     REASONER_BACKEND=vllm NEO4J_URI=bolt://localhost:7687 \\"
echo "       VLLM_BASE=http://localhost:8000/v1 uvicorn api.main:app --port 8080 &"
echo "     streamlit run ui/app.py --server.port 8501"
