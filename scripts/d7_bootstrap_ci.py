#!/usr/bin/env python3
"""
ReTraceRx Phase D -- D7: Statistical rigor (bootstrap confidence intervals).

Two parts:
  (1) PER-CELL CIs  [buildable NOW]: for every (ranker, split, candidate_mode, stratum) cell, bootstrap
      the PER-QUERY values (resample held-out pairs with replacement, recompute the metric, take the
      2.5-97.5 percentiles). Bootstrapping the per-query values -- NOT the average -- is the whole point.
  (2) PAIRED delta  [machinery NOW, used vs the agent POST-C]: bootstrap the per-query difference in
      reciprocal rank between two rankers on the SAME queries; a 95% CI excluding 0 => the gap is
      unlikely to be noise. Demonstrated here on kge-vs-mock; the real use is grounded-agent-vs-baseline.

Terms (first use):
  bootstrap   resample the queries with replacement many times to estimate a metric's sampling spread.
  95% CI      the [2.5, 97.5] percentile interval of the bootstrap distribution.
  reciprocal rank (RR)  1/rank for a query (0 if the target was never ranked); mean RR = MRR.
  paired test we resample the SAME query indices for both systems each iteration, so the difference
              isolates the system effect from which queries happened to be sampled.

Reads the D4 per-query long tables (schema: ranker,split,candidate_mode,disease,drug,stratum,rank).

Run:
  python scripts/d7_bootstrap_ci.py                       # all cells + kge-vs-mock demo
  python scripts/d7_bootstrap_ci.py --paired kge_rotate grounded_agent   # post-C
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import polars as pl

D4 = Path("artifacts/d4_harness")
OUT = Path("artifacts/d7_stats")
N_BOOT = 2000
SEED = 20260727


def load_perquery() -> pl.DataFrame:
    files = sorted(glob.glob(str(D4 / "d4_perquery_*.csv")))
    if not files:
        raise FileNotFoundError(f"No D4 per-query tables in {D4}. Run d4_harness.py first.")
    df = pl.concat([pl.read_csv(f) for f in files]).unique()
    # per-query scalars: RR and hit indicators (a miss -> rank null -> RR 0, hit 0)
    return df.with_columns([
        pl.when(pl.col("rank").is_not_null()).then(1.0 / pl.col("rank")).otherwise(0.0).alias("rr"),
        *[pl.when(pl.col("rank").is_not_null() & (pl.col("rank") <= k)).then(1.0).otherwise(0.0)
            .alias(f"hit{k}") for k in (1, 3, 10)],
    ])


def boot_means(cols: dict[str, np.ndarray], n_boot=N_BOOT, seed=SEED) -> dict:
    """Bootstrap the mean of several per-query vectors with ONE shared resample index per iteration."""
    n = len(next(iter(cols.values())))
    if n == 0:
        return {k: dict(point=None, lo=None, hi=None, n=0) for k in cols}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in cols}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, arr in cols.items():
            draws[k].append(arr[idx].mean())
    res = {}
    for k, arr in cols.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        res[k] = dict(point=round(float(arr.mean()), 4), lo=round(float(lo), 4),
                      hi=round(float(hi), 4), n=int(n))
    return res


def per_cell(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    # A, B, and the AB union (reachable-scored) get CIs; C is a ceiling, not scored
    strata_defs = {
        "A_independent": pl.col("stratum") == "A_independent",
        "B_circular":    pl.col("stratum") == "B_circular",
        "AB_reachable":  pl.col("stratum").is_in(["A_independent", "B_circular"]),
    }
    for (ranker, split, mode), g in df.group_by(["ranker", "split", "candidate_mode"]):
        for sname, pred in strata_defs.items():
            sub = g.filter(pred)
            cols = {m: sub[m].to_numpy() for m in ("rr", "hit1", "hit3", "hit10")}
            res = boot_means(cols)
            for metric, key in (("MRR", "rr"), ("Hits@1", "hit1"), ("Hits@3", "hit3"), ("Hits@10", "hit10")):
                r = res[key]
                rows.append(dict(ranker=ranker, split=split, candidate_mode=mode, stratum=sname,
                                 metric=metric, point=r["point"], lo=r["lo"], hi=r["hi"], n=r["n"]))
    return pl.DataFrame(rows)


def paired_delta(df: pl.DataFrame, ranker_a: str, ranker_b: str,
                 stratum="A_independent", n_boot=N_BOOT, seed=SEED) -> list[dict]:
    """Per-query RR(ranker_a) - RR(ranker_b) on the SAME queries; CI excluding 0 => significant."""
    key = ["split", "candidate_mode", "disease", "drug"]
    A = df.filter((pl.col("ranker") == ranker_a) & (pl.col("stratum") == stratum)).select(*key, rr_a="rr")
    B = df.filter((pl.col("ranker") == ranker_b) & (pl.col("stratum") == stratum)).select(*key, rr_b="rr")
    m = A.join(B, on=key, how="inner")
    out = []
    for (split, mode), g in m.group_by(["split", "candidate_mode"]):
        d = (g["rr_a"] - g["rr_b"]).to_numpy()
        n = len(d)
        if n == 0:
            continue
        rng = np.random.default_rng(seed)
        draws = [d[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out.append(dict(a=ranker_a, b=ranker_b, split=split, candidate_mode=mode, stratum=stratum,
                        n_paired=int(n), mean_delta_RR=round(float(d.mean()), 4),
                        lo=round(float(lo), 4), hi=round(float(hi), 4),
                        significant=bool(lo > 0 or hi < 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", nargs=2, metavar=("RANKER_A", "RANKER_B"), default=None,
                    help="bootstrap paired RR delta between two rankers (default: kge_rotate vs mock if present)")
    ap.add_argument("--stratum", default="A_independent")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_perquery()
    rankers = sorted(df["ranker"].unique().to_list())
    print(f"loaded per-query rows for rankers={rankers}")

    # (1) per-cell CIs
    cells = per_cell(df)
    cells.write_csv(OUT / "d7_percell_ci.csv")
    print("\n=== per-cell 95% CIs (Stratum A, MRR & Hits@10) ===")
    show = cells.filter((pl.col("stratum") == "A_independent") & pl.col("metric").is_in(["MRR", "Hits@10"]))
    for r in show.sort(["ranker", "split", "candidate_mode", "metric"]).iter_rows(named=True):
        print(f"  {r['ranker']:12} {r['split']:12} {r['candidate_mode']:10} {r['metric']:8} "
              f"n={r['n']:>3}  {r['point']}  95%CI[{r['lo']}, {r['hi']}]")

    # (2) paired delta -- demo on kge vs mock; real use is vs grounded_agent post-C
    a, b = args.paired if args.paired else ("kge_rotate", "mock")
    if a in rankers and b in rankers:
        pd = paired_delta(df, a, b, stratum=args.stratum)
        json.dump(pd, open(OUT / f"d7_paired_{a}_vs_{b}.json", "w"), indent=2)
        print(f"\n=== paired RR delta: {a} - {b}  (stratum {args.stratum}) ===")
        for r in pd:
            verdict = "SIGNIFICANT (CI excludes 0)" if r["significant"] else "not significant (CI includes 0)"
            print(f"  {r['split']:12} {r['candidate_mode']:10} n={r['n_paired']:>3}  "
                  f"deltaRR={r['mean_delta_RR']}  95%CI[{r['lo']}, {r['hi']}]  {verdict}")
    else:
        print(f"\n[paired] '{a}' and/or '{b}' not in per-query tables -- "
              f"available: {rankers}. (grounded_agent arrives post-Phase-C.)")

    print(f"\nD7 complete -> {OUT}.")


if __name__ == "__main__":
    main()
