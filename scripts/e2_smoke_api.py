#!/usr/bin/env python3
"""
e2_smoke_api.py — smoke-test a RUNNING e2_api server (mirrors c*_smoke pattern).

Start the server first (separate shell, serve_env sourced):
    uvicorn e2_api:app --host 127.0.0.1 --port 8080
Then:
    python scripts/e2_smoke_api.py
    python scripts/e2_smoke_api.py --disease "alzheimer disease" --mode plus_true

Checks (against the REAL shapes):
  * /health ok
  * /link returns a query_disease block (graph_id or alternatives)
  * grounded /repurpose → {ranking:[…]} OR {abstain,…} OR {needs_disambiguation,…}
  * ungrounded /repurpose → {picks:[…] / ranked_drug_ids:[…]} (hallucination expected)
Non-zero exit on any hard failure.
"""
from __future__ import annotations
import argparse, json, sys
import httpx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--disease", default="type 2 diabetes")
    ap.add_argument("--mode", default="whitelist", choices=["whitelist", "plus_true"])
    args = ap.parse_args()

    fails = 0
    def check(name, ok, detail=""):
        nonlocal fails
        print(f"[{'PASS' if ok else 'FAIL':4}] {name}: {detail}")
        if not ok:
            fails += 1

    # 1. health
    try:
        h = httpx.get(f"{args.base}/health", timeout=30).json()
        check("health ok", bool(h.get("ok")), json.dumps(h)[:160])
        if h.get("warnings"):
            print("       warnings:", h["warnings"])
    except Exception as e:
        check("health reachable", False, e); print("\nis the server up?"); sys.exit(1)

    # 2. link
    gid = None
    try:
        q = httpx.post(f"{args.base}/link", timeout=60,
                       json={"disease_text": args.disease}).json()
        gid = q.get("graph_id")
        check("link returns query_disease", ("graph_id" in q) or ("alternatives" in q),
              f"graph_id={gid} link_score={q.get('link_score')} "
              f"alts={len(q.get('alternatives', []))}")
    except Exception as e:
        check("link call", False, e)

    # 3. grounded
    try:
        g = httpx.post(f"{args.base}/repurpose", timeout=300,
                       json={"disease_text": args.disease, "mode": args.mode,
                             "grounded": True}).json()
        if g.get("needs_disambiguation"):
            check("grounded responds (disambiguation)", True,
                  f"{len(g.get('query_disease', {}).get('alternatives', []))} alternatives")
        elif g.get("abstain"):
            check("grounded responds (abstain)", True, f"reason={g.get('abstain_reason') or g.get('reason')}")
        else:
            ranking = g.get("ranking") or g.get("candidates") or []
            check("grounded returns ranking", bool(ranking),
                  f"{len(ranking)} candidates; latency {g.get('_meta', {}).get('latency_s')}s")
            if ranking:
                top = ranking[0]
                keys = [k for k in ("drug_name", "drug_graph_id", "cited_path_ids",
                                    "confidence", "supported", "paths") if k in top]
                check("top candidate carries citation fields", len(keys) >= 2, keys)
    except Exception as e:
        check("grounded call", False, e)

    # 4. ungrounded (contrast; hallucination expected by construction)
    try:
        u = httpx.post(f"{args.base}/repurpose", timeout=300,
                       json={"disease_text": args.disease, "mode": args.mode,
                             "grounded": False}).json()
        picks = u.get("picks") or u.get("ranked_drug_ids") or []
        check("ungrounded returns picks", bool(picks),
              f"{len(picks)} picks; latency {u.get('_meta', {}).get('latency_s')}s")
    except Exception as e:
        check("ungrounded call", False, e)

    print("\n" + ("SMOKE OK ✅" if fails == 0 else f"SMOKE FAILED ❌ ({fails})"))
    sys.exit(0 if fails == 0 else 1)

if __name__ == "__main__":
    main()
