"""
c0_config.py — single source of truth for Phase C paths, splits, modes, the
core reasoner handle, and the two derived-not-hardcoded facts that keep biting
this project.

SCOPE: Core Phase C. The reasoner runs via plain `transformers` (FP16) with
guided-JSON decoding. Serving optimization (vLLM) and INT4/AWQ quantization are
Stretch S1; the TxGemma property tool is a stretch ablation. Their config lives
under the clearly-marked "STRETCH" block at the bottom (used only by the parked
scripts in phaseC/stretch/), never by the core path.

Grounded in the frozen status records:
  - Reachable eval whitelist = 1,565  (Phase A A8/D35).
  - Splits random / disease_cold / drug_cold; disease_cold is the headline split
    (Phase D C-D1: LARGEST reachability ceiling, 23).
  - Modes plus_true / whitelist are DIFFERENT QUESTIONS; never average them
    (Phase D D-D14 / C-D5).
  - Indication-relation casing: KGE/PyKEEN uses a HYPHEN (DRG-DIS__INDICATION),
    Neo4j uses an UNDERSCORE (DRG_DIS__INDICATION). Derive per-consumer; never
    hardcode either literal (Phase D D-D2 / C-D2, Phase B E-B10).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

# --- repo root -------------------------------------------------------------
REPO_ROOT = Path(os.environ.get("RETRACERX_ROOT", os.getcwd())).resolve()

# --- consumed (frozen, read-only) inputs -----------------------------------
DOCS = REPO_ROOT / "docs"
DATA = REPO_ROOT / "data"
SCRIPTS = REPO_ROOT / "scripts"
ARTIFACTS = REPO_ROOT / "artifacts"
LOGS = REPO_ROOT / "logs"

CONTRACT_MD = DOCS / "retrieval_contract_v1.md"
CONTRACT_SCHEMA = DOCS / "retrieval_contract_v1.schema.json"

WHITELIST_REACHABLE = DATA / "holdout" / "candidate_whitelist_reachable.csv"
WHITELIST_REACHABLE_N = 1565
CANON_NODES = DATA / "canon" / "canon_nodes.parquet"   # SMILES lookup (stretch property tool)

def kge_model_pkl(split: str) -> Path:
    return ARTIFACTS / f"kge_rotate_{split}" / "trained_model.pkl"

def d1_meta_json(split: str) -> Path:
    return ARTIFACTS / "kge_data" / split / "d1_meta.json"

# Phase B shared code the verifier and agent import (never reimplement).
B5_SERIALIZE = SCRIPTS / "b5_serialize.py"     # compute_path_id (D-B18)
B6_TOOLS = SCRIPTS / "b6_tools.py"             # link/retrieve/score/link_and_retrieve
D4_HARNESS = SCRIPTS / "d4_harness.py"         # evaluate(ranker, split, mode) (D-D16/17)
NEO4J_ENV_SH = SCRIPTS / "neo4j_env.sh"

B6_CALL_LOG = LOGS / "b6_calls.jsonl"          # tools append here (D-B21)

# --- Phase C produced outputs ----------------------------------------------
PHASEC_DIR = DATA / "phaseC"
C0_PREFLIGHT_JSON = PHASEC_DIR / "c0_preflight.json"

# --- splits / modes --------------------------------------------------------
SPLITS = ["random", "disease_cold", "drug_cold"]
REQUIRED_SPLITS = ["random", "disease_cold"]
HEADLINE_SPLIT = "disease_cold"
MODES = ["plus_true", "whitelist"]

# --- CORE reasoner: Qwen 3.5-9B via transformers (FP16), guided JSON --------
# No server, no quantization (those are Stretch S1). The agent imports the
# reasoner in-process from .venv-data (torch + transformers already pinned).
# CONFIRM the exact Hugging Face repo id for Qwen 3.5-9B before loading.
REASONER_MODEL_REPO = os.environ.get("REASONER_MODEL_REPO", "Qwen/Qwen3.5-9B")
REASONER_DTYPE = os.environ.get("REASONER_DTYPE", "float16")   # FP16 ~18 GB, fits the 24 GB L4
REASONER_DEVICE_MAP = os.environ.get("REASONER_DEVICE_MAP", "auto")
REASONER_SEED = 20260727                # project split seed, reused for reproducibility
REASONER_MAX_NEW_TOKENS = int(os.environ.get("REASONER_MAX_NEW_TOKENS", "512"))
# Guided-JSON backend for transformers. "lmfe" (lm-format-enforcer) has the most
# stable transformers logits-processor API; "outlines" and "xgrammar" are the
# plan-named alternatives (all three are on vLLM's approved backend list too).
GUIDED_BACKEND = os.environ.get("GUIDED_BACKEND", "xgrammar")  # xgrammar | lmfe | outlines

# --- derived, never-hardcoded indication casing (D-D2 / C-D2) --------------
def read_kge_ind_relation(split: str) -> str:
    """The HYPHEN composite string as PyKEEN/KGE stores it, read from the file."""
    meta = json.loads(d1_meta_json(split).read_text())
    ind = meta.get("ind_relation")
    if not ind:
        raise KeyError(f"'ind_relation' missing from {d1_meta_json(split)}")
    return ind  # e.g. 'DRG-DIS__INDICATION'

def to_neo4j_type(kge_composite: str) -> str:
    """KGE composite (hyphen label) -> Neo4j TYPE (underscore label). Only the
    label hyphen differs; the relation half already uses underscores."""
    return kge_composite.replace("-", "_")

# =====================================================================
# STRETCH config (used ONLY by phaseC/stretch/*, not the core path).
# =====================================================================
# S1 serving (vLLM) knobs live in phaseC/stretch/S1_serving/vllm.env.
# Property-tool (TxGemma) precompute constants:
TXGEMMA_REPO = os.environ.get("TXGEMMA_REPO", "google/txgemma-2b-predict")
TXGEMMA_TOX_TASKS = ["ClinTox", "hERG", "AMES"]
QED_DRUGLIKE_THRESHOLD = 0.5
TXGEMMA_PROPS_PARQUET = PHASEC_DIR / "txgemma_props.parquet"

if __name__ == "__main__":
    print(f"REPO_ROOT        = {REPO_ROOT}")
    print(f"whitelist        = {WHITELIST_REACHABLE}  (expect {WHITELIST_REACHABLE_N})")
    print(f"reasoner repo    = {REASONER_MODEL_REPO}  dtype={REASONER_DTYPE}  (CONFIRM 9B id)")
    print(f"guided backend   = {GUIDED_BACKEND}")
    print(f"phaseC out dir   = {PHASEC_DIR}")

# --- C2 agent knobs --------------------------------------------------------
REASON_TOP_K = int(os.environ.get("REASON_TOP_K", "10"))   # candidates the reasoner ranks (dense JSON: keep modest)
REASON_MAX_NEW_TOKENS = int(os.environ.get("REASON_MAX_NEW_TOKENS", "128"))  # per-candidate DECISION object (all bounded fields)
RATIONALE_MAX_LEN = int(os.environ.get("RATIONALE_MAX_LEN", "160"))  # C7 rationale cap (rationale is NOT in the guided schema; xgrammar ignores maxLength)
LINK_ALT_MIN = float(os.environ.get("LINK_ALT_MIN", "0.5")) # min score to accept a link alternative

# --- reasoner backend: in-process transformers OR vLLM HTTP (same interface) ---
# "vllm" pulls S1 serving forward for the report runs; FP16 (not AWQ) so outputs
# stay numerically consistent with the transformers runs. AWQ remains S1 benchmark.
REASONER_BACKEND = os.environ.get("REASONER_BACKEND", "transformers")   # transformers | vllm
REASONER_BASE_URL = os.environ.get("REASONER_BASE_URL", "http://localhost:8000/v1")
REASONER_SERVED_NAME = os.environ.get("REASONER_SERVED_NAME", "reasoner")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "retracerx-local")
REASONER_HTTP_TIMEOUT = int(os.environ.get("REASONER_HTTP_TIMEOUT", "120"))
REASONER_BATCH_CONCURRENCY = int(os.environ.get("REASONER_BATCH_CONCURRENCY", "16"))

# per-generation audit log (B6 determinism ethos; future-proofs the C-fix list)
REASONER_CALL_LOG = LOGS / "c_reasoner_calls.jsonl"
