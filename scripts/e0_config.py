"""
e0_config.py — Phase E single source of truth for pins, paths, ports, seeds.

Mirrors the c0_config.py pattern (Phase C D-C2): nothing downstream hardcodes a
path, pin, or port — everything is read from here so E1..E9 stay consistent with
the frozen A/B/C/D constraints.

Import-light on purpose (stdlib only) so e0_preflight can run before the serve
venv is fully built.
"""
from __future__ import annotations
import os
from pathlib import Path

# --- repo root ---------------------------------------------------------------
# This file lives at <repo>/scripts/e0_config.py (flat layout, no phaseE subfolder),
# so the repo root is ONE level up. Override with RETRACERX_REPO if ever needed.
REPO = Path(os.environ.get("RETRACERX_REPO", Path(__file__).resolve().parents[1]))

# --- pinned model / determinism (Phase C D-C1, D-C14) ------------------------
REASONER_MODEL_REPO = "Qwen/Qwen3.5-9B"     # instruct, FP16 — no quantization for the demo
REASONER_BACKEND    = "vllm"                 # the demo REQUIRES vllm (Phase C E-C9)
DTYPE               = "float16"              # FP16 forced (Qwen3.5 is bf16-native; keeps parity)
MAX_MODEL_LEN       = 8192                    # bound so weights (~18.1 GB) + KV-cache fit the L4
SPLIT_SEED          = 20260727               # greedy + this seed → byte-identical replay (D-C14)
TEMPERATURE         = 0.0                     # greedy

# --- GPU budget facts (Phase C §4 / E1) --------------------------------------
GPU_TOTAL_MIN_MIB   = 22000                   # L4 reports ~23 GB; require a real 24 GB-class card
WEIGHTS_RESIDENT_GB = 18.1                    # measured FP16 resident (Phase C §4)
KV_HEADROOM_MIN_GB  = 3.0                      # leave at least this for the KV-cache

# --- ports (all reached by the HOST app via localhost published ports, E3) ---
PORTS = {
    "vllm":      8000,   # OpenAI-compatible HTTP  -> http://localhost:8000/v1
    "neo4j_bolt":7687,   # Bolt                    -> bolt://localhost:7687
    "neo4j_http":7474,   # browser
    "api":       8080,   # FastAPI (host venv, E2/E3)
    "ui":        8501,    # Streamlit (host venv, E3)
}

# --- Neo4j (Phase A/B: filtered variant, APOC 5.26.28, stock image) ----------
NEO4J_CONTAINER = "retracerx-neo4j"
NEO4J_URI       = os.environ.get("NEO4J_URI", f"bolt://localhost:{PORTS['neo4j_bolt']}")
NEO4J_USER      = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS      = os.environ.get("NEO4J_PASS", "retracerx")
NEO4J_DB        = os.environ.get("NEO4J_DB", "neo4j")
# A leakage composite that MUST return zero rows on the filtered variant (Phase A A3 gate).
NEO4J_LEAKAGE_PROBE_LABEL = "DRG_DIS__INDICATION"

# --- splits (frozen) ---------------------------------------------------------
SPLITS = ["random", "disease_cold", "drug_cold"]
HEADLINE_SPLIT = "disease_cold"          # D6 headline split (Phase D §C / D-D25)
WHITELIST_REACHABLE_N = 1565

# --- consumed upstream artifacts (read-only) ---------------------------------
CONTRACT_MD     = REPO / "docs" / "retrieval_contract_v1.md"
CONTRACT_SCHEMA = REPO / "docs" / "retrieval_contract_v1.schema.json"
B5_SERIALIZE    = REPO / "scripts" / "b5_serialize.py"      # compute_path_id (shared, D-B18)
B6_TOOLS        = REPO / "scripts" / "b6_tools.py"          # link_entity/retrieve_candidates/...
NEO4J_ENV_SH    = REPO / "scripts" / "neo4j_env.sh"         # start+wait+export+ping (Phase B)
WHITELIST_CSV   = REPO / "data" / "holdout" / "candidate_whitelist_reachable.csv"

def kge_model(split: str) -> Path:
    return REPO / "artifacts" / f"kge_rotate_{split}" / "trained_model.pkl"

def kge_meta(split: str) -> Path:
    return REPO / "artifacts" / "kge_data" / split / "d1_meta.json"

def split_positives(split: str) -> Path:
    return REPO / "data" / "holdout" / "splits" / f"{split}_holdout_positives.csv"

# --- Phase C serving artifacts to REUSE (do not re-author, E3 watch-out) ------
# Flat repo: the FP16 serve script lives in scripts/. vllm.env may sit next to it;
# if it doesn't exist, the pinned digest is grepped from the serve script itself.
FP16_SERVE_SH    = REPO / "scripts" / "c1_serve_vllm_fp16.sh"   # canonical FP16 serve command
VLLM_ENV_CANDIDATES = [
    REPO / "scripts" / "vllm.env",
    REPO / "vllm.env",
    REPO / "phaseC" / "stretch" / "S1_serving" / "vllm.env",    # legacy per Phase C doc
]

def resolve_vllm_image():
    """Return (image_ref, source_path) for the DIGEST-pinned vLLM image, or (None, None).
    Looks in any vllm.env candidate first, then falls back to the FP16 serve script."""
    import re
    pat = re.compile(r"vllm/vllm-openai@sha256:[0-9a-f]{64}")
    for env in VLLM_ENV_CANDIDATES:
        if env.exists():
            m = pat.search(env.read_text(errors="ignore"))
            if m:
                return m.group(0), env
    if FP16_SERVE_SH.exists():
        m = pat.search(FP16_SERVE_SH.read_text(errors="ignore"))
        if m:
            return m.group(0), FP16_SERVE_SH
    return None, None

# vLLM was served with an API key (via VLLM_API_KEY env), so clients must send
# `Authorization: Bearer <key>`. It is also served under a RENAMED id
# (--served-model-name), so requests use that name, not the repo string.
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")   # explicit env wins

def resolve_vllm_api_key():
    """Return (api_key, source) — env first, then --api-key / VLLM_API_KEY / the
    ${VLLM_API_KEY:-default} shell form in the serve script or a vllm.env.
    (None, None) if the server is open (no auth)."""
    import re
    if VLLM_API_KEY:
        return VLLM_API_KEY, "env:VLLM_API_KEY"
    sources = [p for p in ([FP16_SERVE_SH] + VLLM_ENV_CANDIDATES) if p.exists()]
    pats = [
        r"--api-key(?:[=\s]+)(['\"]?)([^\s'\"]+)\1",     # explicit --api-key flag
        r"VLLM_API_KEY:-([^}\s'\"]+)",                     # ${VLLM_API_KEY:-default}
        r"VLLM_API_KEY\s*=\s*(['\"]?)([^\s'\"]+)\1",      # VLLM_API_KEY=literal
    ]
    for src in sources:
        txt = src.read_text(errors="ignore")
        for p in pats:
            m = re.search(p, txt)
            if m:
                val = m.groups()[-1]                       # last group = the value
                if val and not val.startswith("$"):        # skip unresolved shell vars
                    return val, str(src)
    return None, None

SERVED_MODEL_NAME = os.environ.get("VLLM_SERVED_MODEL_NAME")   # override

def resolve_served_model_name():
    """The API-facing model id (--served-model-name). Prefer env, then the serve
    script, else the observed default 'reasoner'. NOTE: the live /v1/models id is
    the authoritative source — the healthcheck reads it directly."""
    import re
    if SERVED_MODEL_NAME:
        return SERVED_MODEL_NAME
    if FP16_SERVE_SH.exists():
        m = re.search(r"--served-model-name(?:[=\s]+)(['\"]?)([^\s'\"]+)\1",
                      FP16_SERVE_SH.read_text(errors="ignore"))
        if m and not m.group(2).startswith("$"):
            return m.group(2)
    return "reasoner"

# --- Phase D/C dashboard inputs (E5 needs these; DoD-relevant) ---------------
D7_PERCELL_CI    = REPO / "artifacts" / "d7_stats" / "d7_percell_ci.csv"
PAIRED_SUMMARY   = REPO / "data" / "phaseC" / "paired" / "paired_summary.csv"
def c3_faithfulness(split: str) -> Path:
    return REPO / "data" / "phaseC" / f"c3_faithfulness_{split}_plus_true.json"
def c8_calibration(split: str) -> Path:
    return REPO / "data" / "phaseC" / "c8" / f"c8_calibration_{split}.json"

# --- Phase E outputs ---------------------------------------------------------
PHASEE_OUT       = REPO / "data" / "phaseE"
E0_PREFLIGHT_JSON = PHASEE_OUT / "e0_preflight.json"
E1_HEALTH_JSON    = PHASEE_OUT / "e1_healthcheck.json"

# --- serve venv (host, uv) ---------------------------------------------------
SERVE_VENV = REPO / ".venv-serve"

if __name__ == "__main__":
    # quick human dump
    print(f"REPO                = {REPO}")
    print(f"model / backend     = {REASONER_MODEL_REPO} / {REASONER_BACKEND} ({DTYPE})")
    print(f"max_model_len       = {MAX_MODEL_LEN}   seed = {SPLIT_SEED}   temp = {TEMPERATURE}")
    print(f"ports               = {PORTS}")
    print(f"neo4j               = {NEO4J_URI} ({NEO4J_CONTAINER})")
    _img, _src = resolve_vllm_image()
    print(f"vllm image (digest) = {_img or '<not found>'}  (from {_src or 'n/a'})")
    print(f"fp16 serve script   = {FP16_SERVE_SH}")
