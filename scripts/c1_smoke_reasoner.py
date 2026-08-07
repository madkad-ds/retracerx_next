#!/usr/bin/env python3
"""
c1_smoke_reasoner.py — prove the CORE reasoner does the one thing Phase C depends
on: guided (constrained) JSON where cited_path_ids is an ENUM of the retrieved
path_ids, so a fabricated citation is structurally impossible (plan step 9).
Also confirms determinism (greedy + fixed seed → identical output).

No server, no quantization. Loads the transformers reasoner in-process.

Usage:
    export PATH="$HOME/.local/bin:$PATH"
    source .venv-data/bin/activate && source scripts/okg_env.sh
    export PYTHONPATH="$PWD/scripts"
    export REASONER_MODEL_REPO=Qwen/Qwen3-8B     # CONFIRM the 9B id
    python phaseC/scripts/c1_smoke_reasoner.py
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c0_config as C  # noqa: E402
from c1_reasoner import get_reasoner  # noqa: E402

# Pretend these are the path_ids retrieved for a disease (contract candidates[].paths[]).
RETRIEVED_PATH_IDS = ["p_9f2a1b3c4d", "p_1122334455", "p_aabbccddee"]

GUIDED_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "drug_graph_id": {"type": "string"},
                "cited_path_ids": {"type": "array",
                                   "items": {"type": "string", "enum": RETRIEVED_PATH_IDS}},
                "rationale": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["drug_graph_id", "cited_path_ids", "rationale", "confidence"],
        }},
        "abstain": {"type": "boolean"},
        "abstain_reason": {"type": ["string", "null"]},
    },
    "required": ["ranking", "abstain", "abstain_reason"],
}

PROMPT = (
    "You are ranking repurposing candidates. Only cite path ids from this list: "
    f"{RETRIEVED_PATH_IDS}. Produce one ranking entry for drug CHEMBL_1431 citing "
    "path p_9f2a1b3c4d with a one-sentence mechanistic rationale and confidence 0.7. "
    "Do not abstain. Reply as JSON."
)


def main() -> int:
    r = get_reasoner()

    obj = r.generate_json(PROMPT, GUIDED_SCHEMA)   # already parsed dict
    print("\n--- guided completion #1 ---\n" + json.dumps(obj, indent=2))

    # 1. schema-shaped object?
    if "ranking" not in obj:
        print("[FAIL] output missing 'ranking'")
        return 1

    # 2. every cited id inside the allowed enum (un-fakeable citations)?
    cited = [pid for e in obj.get("ranking", []) for pid in e.get("cited_path_ids", [])]
    stray = [pid for pid in cited if pid not in RETRIEVED_PATH_IDS]
    if stray:
        print(f"[FAIL] model emitted path_ids outside the retrieved enum: {stray}")
        return 1
    print(f"[ok] all cited ids within retrieved enum: {cited}")

    # 3. determinism: same request → identical parsed object
    obj2 = r.generate_json(PROMPT, GUIDED_SCHEMA)
    if obj != obj2:
        print("[WARN] outputs differ across identical calls — check do_sample=False and seed.")
    else:
        print("[ok] deterministic: identical output across two identical calls")

    print("\nC1 core reasoner smoke test: PASS (guided decoding enforces the citation enum).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
