#!/usr/bin/env python3
"""
c3_faithfulness_eval.py — FULL-HOLDOUT grounded-vs-ungrounded hallucination gap.

The C3 headline instrument. NOT a recovery eval and NOT a parallel holdout:
recovery comes from the D4 harness per-query CSVs (D-D16). This runs over the
ENTIRE frozen holdout for a split (no --n-diseases sampling), on the same disease
set the harness iterates, and computes only the metric the harness does not:
faithfulness / hallucination.

Per grounded disease, unfaithful picks are split into:
  n_unsupported       -- agent set supported=false           (C8 CALIBRATION signal)
  n_citation_failure  -- supported=true but no citation verified
                         (C3 VERIFIER-CATCH; ~0 under guided decoding)

Ungrounded (no graph) scored by the SAME verifier -> hallucination ~1.0 by
construction. Reports the gap with a bootstrap CI over diseases.

Runs a SAMPLED graph-integrity assertion (c3_verify.sampled_graph_assert) every
run: re-checks a fraction of retrieved path edges exist in Neo4j (undirected).

Leads with disease_cold (the honest zero-shot split).

Usage:
    python scripts/c3_faithfulness_eval.py --split disease_cold          # full holdout
    python scripts/c3_faithfulness_eval.py --split disease_cold --limit 10   # smoke only
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c0_config as C  # noqa: E402
import b6_tools as b6  # noqa: E402
from c2_agent import run_agent  # noqa: E402
from c6_ungrounded import run_ungrounded  # noqa: E402
from c3_verify import score_system, sampled_graph_assert  # noqa: E402

DISEASE_ID_COLS = ["okg_disease_id", "disease_graph_id", "disease", "okg_dis_id", "graph_id"]
SPOT_DISEASES = 5           # diseases retrieved for the sampled graph-integrity assertion


def holdout_diseases(split: str):
    df = pl.read_csv(C.DATA / "holdout" / "splits" / f"{split}_holdout_positives.csv")
    col = next((c for c in DISEASE_ID_COLS if c in df.columns), df.columns[0])
    return df.get_column(col).cast(pl.Utf8).unique().sort().to_list()   # ALL, deterministic


def bootstrap_ci(vals, iters=10000, seed=20260727, alpha=0.05):
    if not vals:
        return (None, None)
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (means[int((alpha / 2) * iters)], means[int((1 - alpha / 2) * iters)])


def graph_integrity_spotcheck(diseases, split, mode):
    """Retrieve a few diseases' contracts and assert a sample of their path edges
    exist in the graph. Fail fast before the full loop."""
    sample = diseases[:SPOT_DISEASES]
    paths = []
    for did in sample:
        try:
            contract = b6.retrieve_candidates(did, split=split) \
                if "split" in b6.retrieve_candidates.__code__.co_varnames \
                else b6.retrieve_candidates(did)
            cands = contract["candidates"] if isinstance(contract, dict) else contract
            for c in cands:
                paths.extend(c.get("paths", []))
        except Exception as e:  # noqa: BLE001
            print(f"[spotcheck] retrieve failed for {did}: {e}")
    n = sampled_graph_assert(paths, sample_rate=0.25)
    print(f"[spotcheck] graph-integrity: {n} sampled path(s) verified in-graph "
          f"(undirected) across {len(sample)} diseases\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=C.HEADLINE_SPLIT)   # disease_cold leads
    ap.add_argument("--mode", default="plus_true")
    ap.add_argument("--n-ungrounded", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="smoke only; default = full holdout")
    a = ap.parse_args()

    diseases = holdout_diseases(a.split)
    if a.limit:
        diseases = diseases[: a.limit]
        print(f"[eval] SMOKE: {len(diseases)} of the holdout (use no --limit for the report)\n")
    else:
        print(f"[eval] FULL holdout: {len(diseases)} diseases from {a.split}\n")

    graph_integrity_spotcheck(diseases, a.split, a.mode)

    rows = []
    g_hall_per_disease = []
    tot = dict(n_picks=0, n_faithful=0, n_unsupported=0, n_citation_failure=0, n_abstain=0)
    for did in diseases:
        g = run_agent(disease_graph_id=did, split=a.split, mode=a.mode)
        if g.get("abstain"):
            tot["n_abstain"] += 1
            grounded = {"abstain": True, "n_picks": 0}
        else:
            r = g.get("ranking", [])
            n_picks = len(r)
            n_faith = sum(1 for e in r if e["faithful"])
            n_unsup = sum(1 for e in r if not e["supported"])
            n_citfail = sum(1 for e in r if e["supported"] and not e["faithful"])
            tot["n_picks"] += n_picks; tot["n_faithful"] += n_faith
            tot["n_unsupported"] += n_unsup; tot["n_citation_failure"] += n_citfail
            hall = 0.0 if n_picks == 0 else (n_picks - n_faith) / n_picks
            if n_picks:
                g_hall_per_disease.append(hall)
            grounded = {"abstain": False, "n_picks": n_picks, "n_faithful": n_faith,
                        "n_unsupported": n_unsup, "n_citation_failure": n_citfail,
                        "pick_level_hallucination": round(hall, 4)}

        u = run_ungrounded(did, a.n_ungrounded)
        u_rep = score_system(u["picks"], retrieved_paths=[]).as_dict()

        rows.append({"disease": did, "grounded": grounded,
                     "ungrounded_free": {"n_mapped": u["n_mapped"], **u_rep}})

    n_eval = len(g_hall_per_disease)
    g_mean = sum(g_hall_per_disease) / n_eval if n_eval else None
    lo, hi = bootstrap_ci(g_hall_per_disease)
    summary = {
        "split": a.split, "mode": a.mode, "n_diseases": len(diseases),
        "n_abstain": tot["n_abstain"],
        "grounded": {
            "mean_pick_hallucination": round(g_mean, 4) if g_mean is not None else None,
            "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "totals": tot,
            "unfaithful_breakdown": {
                "unsupported_calibration": tot["n_unsupported"],
                "citation_failure_verifier_catch": tot["n_citation_failure"],
            },
        },
        "ungrounded_free": {"mean_pick_hallucination": 1.0, "note": "by construction (no paths)"},
        "hallucination_gap": round(1.0 - g_mean, 4) if g_mean is not None else None,
        "per_disease": rows,
    }
    C.PHASEC_DIR.mkdir(parents=True, exist_ok=True)
    out = C.PHASEC_DIR / f"c3_faithfulness_{a.split}_{a.mode}.json"
    out.write_text(json.dumps(summary, indent=2))

    print("================= C3 faithfulness (full holdout) =================")
    print(f"split={a.split}  diseases={len(diseases)}  abstained={tot['n_abstain']}")
    print(f"grounded   mean pick-hallucination = {g_mean:.4f}  "
          f"CI95 [{lo:.4f}, {hi:.4f}]  (over {n_eval} non-abstain diseases)")
    print(f"ungrounded mean pick-hallucination = 1.0000  (by construction)")
    print(f"GAP = {1.0 - g_mean:.4f}")
    print(f"\nunfaithful breakdown (grounded): "
          f"unsupported(calibration)={tot['n_unsupported']}  "
          f"citation_failure(verifier-catch)={tot['n_citation_failure']}  "
          f"of {tot['n_picks']} picks ({tot['n_faithful']} faithful)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
