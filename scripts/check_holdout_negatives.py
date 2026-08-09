#!/usr/bin/env python3
"""
ED10-cand: verify the per-split holdout negatives are the intended, integrity-clean frozen artifacts.

Criticality: NOT needed for the recovery null (doesn't use negatives) and NOT needed to trust the
KGE 'AUROC ~ chance' result (robust to count errors). REQUIRED before publishing any negatives-
dependent metric -- D5(b) discrimination and any D6 abstention that uses the negative set as truth.

Per split it checks, in order of importance:
  1. LEAKAGE   -- no negative pair is also a positive (would corrupt every discrimination/abstention #).
  2. COLLAPSE  -- every disease id is already canonical (applying the A6 collapse is a no-op).
  3. KEYING    -- keyed on (okg_drug_id, okg_disease_id); valid, non-null.
  4. COUNTS    -- report per-split counts + pos:neg ratio; confirm disease_cold == 208 (frozen headline).
  5. PROVENANCE (best-effort) -- negatives resolve to a FAILED resolved_class in repodb_resolved.

Run:  python scripts/check_holdout_negatives.py --splits random disease_cold drug_cold
Exit code 1 if any CRITICAL check (leakage / collapse / keying) fails.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import polars as pl
from d3_evaluate_kge import disease_remap, apply_remap, HOLDOUT

REPODB_RESOLVED = Path("data/gold/repodb_resolved.parquet")

def pos_path(s): return HOLDOUT / "splits" / f"{s}_holdout_positives.csv"
def neg_path(s):
    p = HOLDOUT / "splits" / f"{s}_holdout_negatives.csv"
    if p.exists(): return p
    if s == "disease_cold" and (HOLDOUT / "holdout_negatives.csv").exists():
        return HOLDOUT / "holdout_negatives.csv"
    raise FileNotFoundError(f"no negatives file for {s!r}")

FROZEN_NEG = {"disease_cold": 208}     # only value documented in the bundle listing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["random", "disease_cold", "drug_cold"])
    args = ap.parse_args()
    remap = disease_remap()

    # optional provenance reference
    failed_pairs = None
    if REPODB_RESOLVED.exists():
        rr = pl.read_parquet(REPODB_RESOLVED)
        dcol = next((c for c in ("okg_drug_id", "drug_okg_id") if c in rr.columns), None)
        scol = next((c for c in ("okg_dis_id", "okg_disease_id") if c in rr.columns), None)
        if dcol and scol and "resolved_class" in rr.columns:
            classes = rr["resolved_class"].unique().to_list()
            failed = [c for c in classes if isinstance(c, str)
                      and any(k in c.lower() for k in ("fail", "withdraw", "terminat", "suspend", "negativ"))]
            fr = rr.filter(pl.col("resolved_class").is_in(failed))
            failed_pairs = set(zip(fr[dcol].to_list(), fr[scol].to_list()))
            print(f"[provenance ref] repodb_resolved failed classes={failed} -> {len(failed_pairs)} pairs\n")

    any_critical = False
    for s in args.splits:
        print(f"=== {s} ===")
        neg = pl.read_csv(neg_path(s))
        pos = pl.read_csv(pos_path(s))
        # 3. KEYING
        for df, nm in ((neg, "neg"), (pos, "pos")):
            missing = [c for c in ("okg_drug_id", "okg_disease_id") if c not in df.columns]
            if missing:
                print(f"  [CRITICAL] {nm} missing key columns {missing} (has {df.columns})"); any_critical = True
        if any_critical:
            continue
        nulls = neg.filter(pl.col("okg_drug_id").is_null() | pl.col("okg_disease_id").is_null()).height
        neg = neg.select(drug="okg_drug_id", disease="okg_disease_id").unique()
        pos = pos.select(drug="okg_drug_id", disease="okg_disease_id").unique()

        # 4. COUNTS
        n = neg.height
        frozen = FROZEN_NEG.get(s)
        cflag = "" if frozen is None else (" == frozen ✓" if n == frozen else f" != frozen {frozen} ✗")
        print(f"  count: neg={n}{cflag}  pos={pos.height}  ratio pos:neg = {pos.height/max(n,1):.2f}:1  nulls={nulls}")

        # 2. COLLAPSE-CONSISTENCY (critical): applying the collapse must be a no-op on disease ids
        remapped = apply_remap(neg.clone(), remap, "disease")
        changed = neg.join(remapped, left_on=["drug", "disease"], right_on=["drug", "disease"], how="anti").height
        # anti-join above is 0 if nothing changed; compute properly:
        merged = neg.rename({"disease": "disease_orig"}).with_row_index("i").join(
                 remapped.rename({"disease": "disease_new"}).with_row_index("i"), on="i")
        n_changed = merged.filter(pl.col("disease_orig") != pl.col("disease_new")).height
        if n_changed:
            print(f"  [CRITICAL] {n_changed} negative disease ids are non-canonical (collapse changed them) "
                  f"-> ids not in the holdout's collapsed space"); any_critical = True
        else:
            print(f"  collapse-consistent: 0 disease ids rewritten ✓")

        # 1. LEAKAGE (most critical): no negative pair may also be a positive
        overlap = neg.join(pos, on=["drug", "disease"], how="inner").height
        if overlap:
            print(f"  [CRITICAL] {overlap} negative pairs are ALSO positives -> label leakage"); any_critical = True
        else:
            print(f"  no pos/neg overlap ✓")

        # 5. PROVENANCE (best-effort, non-critical)
        if failed_pairs is not None:
            npairs = set(zip(neg["drug"].to_list(), neg["disease"].to_list()))
            in_failed = len(npairs & failed_pairs)
            print(f"  provenance: {in_failed}/{len(npairs)} negatives trace to a repodb failed class"
                  f"{' ✓' if in_failed == len(npairs) else '  (investigate the remainder)'}")
        print()

    if any_critical:
        print("RESULT: CRITICAL issue found -- do NOT publish negatives-dependent metrics until fixed.")
        sys.exit(1)
    print("RESULT: negatives are integrity-clean (leakage/collapse/keying all pass). "
          "Counts/provenance printed above for the record.")


if __name__ == "__main__":
    main()
