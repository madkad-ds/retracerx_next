#!/usr/bin/env bash
# e0_bootstrap.sh — build the HOST "env-serve" venv for the FastAPI + Streamlit app.
#
# The env-isolation rule (E1/E3): the finicky GPU/DB services run in containers;
# the light app runs on the host from a `uv` venv (uv = a fast Python package/venv
# manager) and talks to them over published localhost ports. This script builds
# that venv and, critically, verifies the frozen torch pin is NOT disturbed
# (Phase C carried torch 2.13.0+cu130 through every add).
#
# Run from the repo root:  bash scripts/e0_bootstrap.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
VENV="$REPO/.venv-serve"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH. Install uv first."; exit 1; }

echo "==> Creating serve venv at $VENV"
uv venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Record the torch pin BEFORE we install anything, to constrain later installs.
# (Phase C C1-3: +cu130 is a local version tag; pin via a constraint file, don't reinstall.)
PHASEC_LOCK="$REPO/requirements-phaseC.lock"
if [[ ! -f "$PHASEC_LOCK" ]]; then
  echo "ERROR: $PHASEC_LOCK missing (needed so agent/verifier imports resolve)."; exit 1
fi

echo "==> Installing pinned Phase C runtime (agent, verifier, tools) from lock"
# Install the frozen Phase C lock so c_agent / c_verify / b6_tools import cleanly.
uv pip install -r "$PHASEC_LOCK"

# Capture the torch pin now present, use it as a hard constraint for the serving extras.
TORCH_CONSTRAINT="$(mktemp)"
uv pip freeze | grep -E '^torch==' > "$TORCH_CONSTRAINT" || true
echo "==> torch pin held at: $(cat "$TORCH_CONSTRAINT" || echo '<none found>')"

echo "==> Installing serving/UI extras (host app only) under the torch constraint"
# fastapi/uvicorn  -> the API (E2);  streamlit/pyvis -> the UI (E4);
# neo4j            -> Bolt driver;   openai          -> vLLM OpenAI-compatible client;
# pydantic/jsonschema -> request + contract validation.
uv pip install --constraint "$TORCH_CONSTRAINT" \
  "fastapi" "uvicorn[standard]" "streamlit" "pyvis" \
  "neo4j" "openai>=1.0" "pydantic>=2" "jsonschema" "httpx"

echo "==> Verifying the torch pin survived the serving installs"
AFTER="$(uv pip freeze | grep -E '^torch==' || true)"
BEFORE="$(cat "$TORCH_CONSTRAINT" || true)"
if [[ -n "$BEFORE" && "$AFTER" != "$BEFORE" ]]; then
  echo "ERROR: torch pin moved!  before='$BEFORE'  after='$AFTER'"; exit 1
fi
rm -f "$TORCH_CONSTRAINT"

echo "==> Freezing the serve lockfile"
uv pip freeze > "$REPO/requirements-serve.lock"

echo "✅ serve venv ready: $VENV"
echo "   activate with:  source $VENV/bin/activate"
