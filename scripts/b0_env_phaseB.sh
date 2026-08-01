#!/usr/bin/env bash
# ReTraceRx Phase B — B0: environment additions for the retrieval linker.
# Adds faiss / rapidfuzz / transformers to the existing .venv-data WITHOUT
# disturbing the pinned GPU torch (2.13.0+CUDA13) that Phase A/D depend on.
set -euo pipefail

# --- session preamble (Phase A §2) ---
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source .venv-data/bin/activate
source scripts/okg_env.sh
export PYTHONPATH="$PWD/scripts"

# --- assert the protected torch BEFORE any install ---
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda?", torch.cuda.is_available())
assert torch.__version__.startswith("2.13"), "Phase A/D pin torch 2.13.0 — abort, do not clobber it"
PY

# --- install additions, pinning torch in the SAME resolve so uv can't move it ---
uv pip install "torch==2.13.0" faiss-cpu rapidfuzz "transformers>=4.40,<5"
# sentence-transformers is OPTIONAL (B2 uses raw transformers). Non-fatal if it balks:
uv pip install sentence-transformers || true
# belt-and-braces: restore the exact pin in case any install nudged it
uv pip install "torch==2.13.0"

# --- re-verify + freeze a SEPARATE Phase-B lock (don't touch requirements-data.lock) ---
python - <<'PY'
import torch, faiss, rapidfuzz, transformers
print("torch", torch.__version__, "| cuda?", torch.cuda.is_available())
print("faiss", getattr(faiss, "__version__", "?"),
      "| rapidfuzz", rapidfuzz.__version__, "| transformers", transformers.__version__)
assert torch.__version__.startswith("2.13"), "torch drifted — reinstall the pin"
PY
uv pip freeze > requirements-phaseB.lock
echo "wrote requirements-phaseB.lock"

# --- optional: warm the SapBERT cache so B2 doesn't first-touch the network mid-run ---
python - <<'PY'
try:
    from transformers import AutoTokenizer, AutoModel
    m = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
    print("SapBERT cached OK")
except Exception as e:
    print("SapBERT warm-up skipped (needs huggingface.co reachable):", e)
PY
