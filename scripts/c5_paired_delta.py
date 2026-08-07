#!/usr/bin/env python3
"""
c5_paired_delta.py — paired significance between two rankers on the SAME held-out
pairs (D-D22: only the paired delta on identical queries has power at Stratum-A
n~=31; single-system point estimates don't).

Consumes two D4 per-query CSVs (schema: ranker,split,candidate_mode,disease,drug,
stratum,rank). Pairs on (split,candidate_mode,disease,drug), filters to a stratum,
and reports:
  * Reciprocal-rank delta (A - B): continuous; per pair 1/rank (0 if unranked/miss).
    Paired bootstrap CI over pairs.
  * Hits@k delta (A - B): binary per pair; McNemar discordance (b = A-hits/B-misses,
    c = B-hits/A-misses) with an exact two-sided binomial p.

Reads the `rank` column as nullable: empty/None = a miss (rr=0, not a hit).

Usage:
    python scripts/c5_paired_delta.py \
        artifacts/d4_harness/d4_perquery_grounded_agent_plus_true.csv \
        artifacts/d4_harness/d4_perquery_kge_plus_true.csv \
        --stratum A_independent --ks 1 3 10 --label-a agent --label-b kge
"""
from __future__ import annotations
import argparse
import math
import random

import polars as pl

KEYS = ["split", "candidate_mode", "disease", "drug"]


def _rr(rank):
    return 1.0 / rank if rank else 0.0            # None/0 -> miss -> 0


def _hit(rank, k):
    return rank is not None and rank <= k


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs (n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(deltas, iters=10000, seed=20260727, alpha=0.05):
    if not deltas:
        return (None, None)
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iters):
        s = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return (lo, hi)


def load(path, stratum, split=None):
    df = pl.read_csv(path, infer_schema_length=10000)
    if stratum != "all":
        df = df.filter(pl.col("stratum") == stratum)
    if split is not None and "split" in df.columns:
        df = df.filter(pl.col("split") == split)
    keys = [k for k in KEYS if k in df.columns]
    # rank -> nullable float (empty string / null both become null)
    df = df.with_columns(pl.col("rank").cast(pl.Float64, strict=False))
    return df.select(keys + ["rank"]), keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--stratum", default="A_independent",
                    help="A_independent | B_circular | all")
    ap.add_argument("--split", default=None,
                    help="restrict to one split (e.g. disease_cold); default = all splits in the file")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 10])
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--bootstrap", type=int, default=10000)
    a = ap.parse_args()

    da, keys_a = load(a.csv_a, a.stratum, a.split)
    db, keys_b = load(a.csv_b, a.stratum, a.split)
    keys = [k for k in KEYS if k in keys_a and k in keys_b]

    j = (da.rename({"rank": "rank_a"})
           .join(db.rename({"rank": "rank_b"}), on=keys, how="inner"))
    rows = j.to_dicts()

    n_pairs = len(rows)
    print(f"paired comparison: A={a.label_a} ({a.csv_a})")
    print(f"                   B={a.label_b} ({a.csv_b})")
    print(f"stratum={a.stratum}  split={a.split or 'ALL'}  join_keys={keys}")
    print(f"matched pairs: {n_pairs}  (A rows={da.height}, B rows={db.height}, "
          f"unmatched dropped A={da.height - n_pairs}, B={db.height - n_pairs})\n")
    if n_pairs == 0:
        print("No matched pairs — check the two files are the same split/mode/stratum.")
        return 1

    # --- reciprocal-rank delta ---
    ra = [r["rank_a"] for r in rows]
    rb = [r["rank_b"] for r in rows]
    deltas = [_rr(x) - _rr(y) for x, y in zip(ra, rb)]
    mean_d = sum(deltas) / n_pairs
    lo, hi = bootstrap_ci(deltas, iters=a.bootstrap)
    mrr_a = sum(_rr(x) for x in ra) / n_pairs
    mrr_b = sum(_rr(y) for y in rb) / n_pairs
    print("Reciprocal-rank (MRR-style):")
    print(f"  {a.label_a} MRR = {mrr_a:.4f}   {a.label_b} MRR = {mrr_b:.4f}")
    print(f"  mean delta (A-B) = {mean_d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"(paired bootstrap n={a.bootstrap})")
    signif = (lo is not None and (lo > 0 or hi < 0))
    print(f"  -> CI {'EXCLUDES' if signif else 'INCLUDES'} 0 "
          f"({'a real paired difference' if signif else 'not distinguishable'})\n")

    # --- Hits@k delta + McNemar ---
    print("Hits@k discordance (McNemar):")
    print(f"  {'k':>3} {'both':>5} {'A_only(b)':>10} {'B_only(c)':>10} {'neither':>8} "
          f"{'Δrate':>8} {'p':>8}")
    for k in a.ks:
        ha = [_hit(x, k) for x in ra]
        hb = [_hit(y, k) for y in rb]
        both = sum(1 for x, y in zip(ha, hb) if x and y)
        b = sum(1 for x, y in zip(ha, hb) if x and not y)     # A hit, B miss
        c = sum(1 for x, y in zip(ha, hb) if y and not x)     # B hit, A miss
        neither = n_pairs - both - b - c
        drate = (b - c) / n_pairs
        p = mcnemar_exact(b, c)
        print(f"  {k:>3} {both:>5} {b:>10} {c:>10} {neither:>8} {drate:>+8.4f} {p:>8.4f}")

    # --- interpretation hint ---
    total_discordant = sum(
        sum(1 for x, y in zip([_hit(v, k) for v in ra], [_hit(v, k) for v in rb]) if x != y)
        for k in a.ks)
    print()
    if total_discordant == 0:
        print("No discordant pairs at any k: the two systems never disagree on a hit. "
              "There is no signal to power here — enlarging the holdout will NOT create it; "
              "the ceiling is retrieval (3-hop is the lever), not sample size.")
    else:
        print("Some discordance exists: if the CI still includes 0, MORE Stratum-A pairs "
              "(holdout enlargement) is the honest lever to reach significance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
