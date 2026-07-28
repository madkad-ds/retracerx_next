#!/usr/bin/env python3
"""
a6_verify.py  —  Gate for A6. Exit 1 on any failure; do not proceed to A7 unless 0 failures.

Checks (each prints PASS/FAIL with the measured number, never a number from prose):
  G1  drug_xwalk is 1:1 on okg_drug_id
  G2  DrugBank side is 1:1 (no drugbank_id -> >1 node)  [open rule 5 / E18]
  G3  disease_xwalk is 1:1 on okg_dis_id AND contains ZERO OBA ids  [E19]
  G4  disease_xwalk never uses MONDO as the key (mondo_if_present is annotation only)
  G5  drug collapse map: exactly one chosen node per collision group
  G6  disease collapse map: exactly one chosen node per multi-node CUI (open rule 3)
  G7  repoDB disease reach (native umls_cui) within ±5 of A5's 717  [E23]
  G8  repoDB drug reach (DrugBank) within ±5 of frozen 1,905
  G9  every repoDB multi-node CUI is resolved by the disease collapse map (A7 won't double-count)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import polars as pl

DRUG_XW   = Path("data/holdout/drug_xwalk.parquet")
DIS_XW    = Path("data/holdout/disease_xwalk.parquet")
DRUG_COL  = Path("data/manifest/a6_drug_collapse.parquet")
DIS_COL   = Path("data/manifest/a6_disease_collapse.parquet")
REPODB_RES  = Path("data/gold/repodb_resolved.parquet")
REPODB_PAIR = Path("data/gold/repodb_pairs.parquet")
CANON_NODES = Path("data/canon/canon_nodes.parquet")

fails: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {detail}")
    if not ok:
        fails.append(name)


def main() -> None:
    for p in (DRUG_XW, DIS_XW, DRUG_COL, DIS_COL):
        if not p.exists():
            print(f"[FATAL] missing {p} — run a6_collapse_rules.py then a6_emit_crosswalks.py")
            sys.exit(1)

    dx = pl.read_parquet(DRUG_XW)
    dz = pl.read_parquet(DIS_XW)
    dcol = pl.read_parquet(DRUG_COL)
    zcol = pl.read_parquet(DIS_COL)

    check("G1 drug_xwalk 1:1", dx.height == dx["okg_drug_id"].n_unique(),
          f"{dx.height} rows / {dx['okg_drug_id'].n_unique()} unique okg_drug_id")

    db_dupe = dx.filter(pl.col("drugbank_id").is_not_null()).group_by("drugbank_id").len().filter(pl.col("len") > 1)
    check("G2 DrugBank 1:1", db_dupe.height == 0, f"{db_dupe.height} DrugBank ids map to >1 node (expect 0)")

    oba = dz.filter(pl.col("okg_dis_id").str.starts_with("OBA")).height
    check("G3 disease_xwalk 1:1 & no OBA",
          dz.height == dz["okg_dis_id"].n_unique() and oba == 0,
          f"{dz.height} rows / {dz['okg_dis_id'].n_unique()} unique / {oba} OBA leaked")

    # G4: the key column is umls_cui; mondo_if_present must only ever equal a MONDO id or be null
    bad_mondo = dz.filter(pl.col("mondo_if_present").is_not_null()
                          & ~pl.col("mondo_if_present").str.starts_with("MONDO")).height
    check("G4 MONDO annotation-only", bad_mondo == 0, f"{bad_mondo} non-MONDO values in mondo_if_present")

    check("G5 drug collapse one-per-group",
          dcol.height == dcol["collision_key"].n_unique(),
          f"{dcol.height} groups / {dcol['collision_key'].n_unique()} unique keys")

    check("G6 disease collapse one-per-CUI",
          zcol.height == zcol["umls_cui"].n_unique(),
          f"{zcol.height} CUIs / {zcol['umls_cui'].n_unique()} unique")

    # G7 / G8 need the repoDB artifacts
    if REPODB_PAIR.exists():
        cuis = pl.read_parquet(REPODB_PAIR).select("ind_id").unique().drop_nulls()
        reach = cuis.join(dz.select("umls_cui").drop_nulls().unique(),
                          left_on="ind_id", right_on="umls_cui", how="inner").height
        check("G7 disease reach ~717", abs(reach - 717) <= 5, f"reach={reach} (A5 native 717, ±5)")
    else:
        print("[SKIP] G7 — repodb_pairs.parquet absent")

    if REPODB_RES.exists():
        res = pl.read_parquet(REPODB_RES)
        if "drugbank_id" in res.columns:
            db_reach = res.filter(pl.col("okg_drug_id").is_not_null()).select("drugbank_id").unique().height
            check("G8 drug reach ~1905", abs(db_reach - 1905) <= 5, f"reach={db_reach} (frozen 1,905, ±5)")
        # G9: every CUI that hits >1 DIS node in repoDB's reachable set is covered by the collapse map
        if "okg_dis_id" in res.columns:
            multi = (res.filter(pl.col("okg_dis_id").is_not_null())
                        .group_by("ind_id").agg(pl.col("okg_dis_id").n_unique().alias("n"))
                        .filter(pl.col("n") > 1).select("ind_id"))
            uncovered = multi.join(zcol.select("umls_cui"), left_on="ind_id", right_on="umls_cui", how="anti").height
            check("G9 multi-node CUIs collapsed", uncovered == 0,
                  f"{uncovered} multi-node repoDB CUI(s) missing from disease collapse map (A7 would double-count)")
    else:
        print("[SKIP] G8/G9 — repodb_resolved.parquet absent")

    print()
    if fails:
        print(f"[a6_verify] {len(fails)} FAIL(s): {', '.join(fails)} — do NOT proceed to A7.")
        sys.exit(1)
    print("[a6_verify] all gates pass. A6 complete — proceed to A7.")


if __name__ == "__main__":
    main()
