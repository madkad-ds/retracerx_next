#!/usr/bin/env python3
"""
ReTraceRx Phase B — B7: retrieval-only recall@K (the system ceiling).

For each held-out positive (drug, disease) in a split, rank candidates for the disease
and check where (if anywhere) the true drug appears. Measured BEFORE the LLM so retrieval
quality is separated from reasoning quality.

Metrics (all reported; each answers a different question):
  * coverage_ceiling   — fraction of true drugs that are even IN the reachable whitelist.
                         A true drug outside it CANNOT be returned (D-D14) — a data
                         property, not a model failure. This is the hard cap on raw recall.
  * pool_recall        — true drug found at ANY rank (did B3 retrieve it at all?). The
                         retriever's own ceiling, independent of B4 ranking.
  * recall@K           — true drug in the top-K after B4 ranking. The deployed metric.
                         pool_recall - recall@K = ranking loss (found but ranked > K).
  Each is reported over ALL positives and over the REACHABLE subset (the retriever's
  actual reach). The reachable-subset recall@K is the honest retrieval-quality number.

Emits a per-query CSV in Phase D's D-D17 long-table schema so D6/D7/D8 can consume it.

NOTE (interpretation): ~94.5% of gold positives are circular (already indication edges,
removed at A7). The meaningful subset is Stratum A (graph-independent). This script reports
overall + reachable; join the per-query CSV to a5_circularity for the Stratum-A cut.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

import polars as pl
import b4_rank

SPLITS_DIR = Path("data/holdout/splits")
REACHABLE = Path("data/holdout/candidate_whitelist_reachable.csv")
EVAL_K_MAX = 10**9        # get the FULL ranked pool (measure true pool recall)
CUTOFFS = [5, 10, 20, 50, 100]


def _col(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return df.columns[0]


def load_positives(split: str):
    f = SPLITS_DIR / f"{split}_holdout_positives.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found — check the split name / holdout bundle.")
    df = pl.read_csv(f)
    dcol = _col(df, "okg_drug_id", "drug_graph_id", "okg_drug")
    scol = _col(df, "okg_disease_id", "okg_disease", "disease_graph_id")
    return [(r[dcol], r[scol]) for r in df.iter_rows(named=True)], (dcol, scol)


def main(split: str, out_prefix: str):
    wl = pl.read_csv(REACHABLE)
    reach = set(wl[_col(wl, "okg_drug_id")].to_list())
    positives, cols = load_positives(split)
    diseases = sorted({d for _, d in positives})
    print(f"split={split}: {len(positives)} positives, {len(diseases)} unique diseases "
          f"| keyed on {cols} | reachable whitelist={len(reach)}")

    # rank each disease once (full pool). A per-disease timeout is a RETRIEVAL OUTCOME
    # (0 candidates for that disease), not a reason to abort the whole eval.
    ranking: dict[str, dict[str, int]] = {}
    timed_out = []
    for i, dis in enumerate(diseases, 1):
        try:
            cands = b4_rank.rank_candidates(dis, split=split, k=EVAL_K_MAX)
            ranking[dis] = {c["drug_graph_id"]: c["rank"] for c in cands}
        except Exception as e:
            ranking[dis] = {}
            timed_out.append(dis)
            print(f"  WARN {dis}: retrieval failed ({type(e).__name__}) -> 0 candidates")
        if i % 25 == 0 or i == len(diseases):
            print(f"  ranked {i}/{len(diseases)} diseases")
    if timed_out:
        print(f"  {len(timed_out)} diseases failed retrieval (timeout/error): {timed_out}")

    rows = []
    for drug, dis in positives:
        rmap = ranking.get(dis, {})
        rank = rmap.get(drug)
        rows.append({"ranker": "graphrag_retrieval", "split": split,
                     "candidate_mode": "whitelist_reachable", "disease": dis, "drug": drug,
                     "reachable": drug in reach, "in_pool": drug in rmap, "rank": rank})

    N = len(rows)
    reach_rows = [r for r in rows if r["reachable"]]
    nR = len(reach_rows)

    def recall_at(K, subset):
        if not subset:
            return None
        return round(sum(1 for r in subset if r["rank"] is not None and r["rank"] <= K)
                     / len(subset), 4)

    hit_ranks = [r["rank"] for r in reach_rows if r["rank"] is not None]
    metrics = {
        "split": split, "positives": N, "unique_diseases": len(diseases),
        "reachable_positives": nR,
        "coverage_ceiling": round(nR / N, 4),           # frac whose true drug is returnable
        "pool_recall_all": round(sum(r["in_pool"] for r in rows) / N, 4),
        "pool_recall_reachable": round(sum(r["in_pool"] for r in reach_rows) / nR, 4) if nR else None,
        "recall_at": {str(K): {"all": recall_at(K, rows),
                               "reachable": recall_at(K, reach_rows)} for K in CUTOFFS},
        "median_hit_rank_reachable": (statistics.median(hit_ranks) if hit_ranks else None),
        "diseases_with_zero_candidates": sum(1 for d in diseases if not ranking[d]),
        "diseases_retrieval_failed": len(timed_out),
        "diseases_retrieval_failed_ids": timed_out,
    }

    Path(f"{out_prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    pl.DataFrame(rows).write_csv(f"{out_prefix}_perquery.csv")

    print("\n=== B7 retrieval recall ===")
    print(f"positives {N} | reachable {nR} ({metrics['coverage_ceiling']:.1%} coverage ceiling)")
    print(f"pool_recall: all={metrics['pool_recall_all']:.3f}  "
          f"reachable={metrics['pool_recall_reachable']}")
    print(f"{'K':>4} | {'recall@K all':>12} | {'recall@K reachable':>18}")
    for K in CUTOFFS:
        a = metrics["recall_at"][str(K)]["all"]
        r = metrics["recall_at"][str(K)]["reachable"]
        print(f"{K:>4} | {a:>12} | {r:>18}")
    print(f"median hit rank (reachable): {metrics['median_hit_rank_reachable']}")
    if metrics["diseases_with_zero_candidates"]:
        print(f"!! {metrics['diseases_with_zero_candidates']} diseases returned 0 candidates "
              "(retrieval gap — check metapath/hub/limit, not the LLM)")
    print(f"\nwrote {out_prefix}_metrics.json and {out_prefix}_perquery.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="disease_cold")
    ap.add_argument("--out-prefix", default=None)
    a = ap.parse_args()
    out = a.out_prefix or f"data/retrieval/b7_{a.split}"
    main(a.split, out)
