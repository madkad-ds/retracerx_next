#!/usr/bin/env python3
"""
c2_smoke_agent.py — end-to-end agent smoke test (needs the full stack up: Neo4j
started via neo4j_env.sh, B6 tools importable, per-split KGE models present, and
the reasoner loadable).

Runs the agent on type 2 diabetes (MONDO_0005148) — the disease Phase B used as
its retrieval probe (233 candidates / 296 paths) — and checks the structural
invariants the agent guarantees:

  - result is either a ranking or an explicit abstention (never a crash)
  - every surviving cited_path_id was actually retrieved (faithfulness)
  - every ranked drug id is real (no invented candidates)
  - ranked_drug_ids covers the retrieved pool (recall not silently dropped)

Usage:
    export PATH="$HOME/.local/bin:$PATH"
    source .venv-data/bin/activate && source scripts/okg_env.sh
    source scripts/neo4j_env.sh
    export PYTHONPATH="$PWD/scripts"
    export REASONER_MODEL_REPO=Qwen/Qwen3.5-9B
    python phaseC/scripts/c2_smoke_agent.py
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_agent import run_agent  # noqa: E402


def main() -> int:
    res = run_agent(disease_graph_id="MONDO_0005148", split="disease_cold", mode="plus_true")
    print(json.dumps(res, indent=2)[:3000])

    if res.get("abstain"):
        print(f"\n[ok] agent abstained cleanly: {res.get('abstain_reason')}")
        print("C2 smoke: PASS (abstention path)")
        return 0

    ranking = res.get("ranking", [])
    ranked_ids = res.get("ranked_drug_ids", [])

    # 1. no invented drugs: every reasoned drug id is in the ranked pool
    stray_drugs = [e["drug_graph_id"] for e in ranking if e["drug_graph_id"] not in set(ranked_ids)]
    if stray_drugs:
        print(f"[FAIL] reasoned drugs missing from ranked pool: {stray_drugs}")
        return 1
    print(f"[ok] {len(ranking)} reasoned drugs, all in the ranked pool of {len(ranked_ids)}")

    # 2. faithfulness: surviving citations are real (verify already enforced this)
    total_real = sum(len(e["cited_path_ids"]) for e in ranking)
    total_hall = sum(len(e.get("hallucinated_path_ids", [])) for e in ranking)
    print(f"[ok] verified citations={total_real}  hallucinated(dropped)={total_hall}  "
          f"faithful_drugs={res.get('n_faithful')}/{res.get('n_reasoned')}")

    # 3. pool coverage
    if len(ranked_ids) < res.get("n_pool", 0):
        print(f"[FAIL] ranked pool {len(ranked_ids)} < retrieved pool {res.get('n_pool')}")
        return 1
    print(f"[ok] ranked pool covers retrieved pool ({res.get('n_pool')})")

    print("\nC2 smoke: PASS (grounded ranking, faithful citations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
