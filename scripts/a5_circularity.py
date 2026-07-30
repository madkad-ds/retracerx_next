#!/usr/bin/env python3
"""
A5 circularity measurement (the substantive A5 task).

repoDB is DrugCentral-built; OKG's DRG-DIS edges carry DrugCentral IDs
(properties.structure_id / drug_disease_id) and sources_direct is 100% populated on DRG-DIS.
So the gold set and the graph's own indication edges are plausibly the SAME upstream data.
A7 removes those edges by construction, so this does not invalidate the design — but it means
"evaluable positive" coverage will look artificially high, and a "novel" recovery may be
recovering an edge the graph already knew via the same source. It MUST be stated in the writeup.

This script measures the overlap and the DrugCentral source share, and writes a5_circularity.json.

Consumes
  data/gold/repodb_resolved.parquet     from a5_gold_set.py (positives resolved to OKG ids)
  data/canon/canon_edges/*.parquet      A2 canonical edges
  data/manifest/a2_integrity.json       expected DRG-DIS/INDICATION edge count (GATE, not prose)
Emits
  data/manifest/a5_circularity.json

Orientation note: DRG-DIS is one of the 13 undirected labels and is stored ONCE per edge in the
source orientation. We therefore compare on a sorted (lo, hi) endpoint tuple rather than (from, to)
or a rebuilt endpoint_key — this is orientation-insensitive and avoids the U+241F separator trap
(rebuilding the key with the wrong separator matches nothing, silently).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import polars as pl

LABEL, RELATION = "DRG-DIS", "INDICATION"


def fatal(msg: str) -> None:
    print(f"[A5-circ FATAL] {msg}", file=sys.stderr); sys.exit(1)


def dig(obj, path, label):
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            here = list(cur.keys()) if isinstance(cur, dict) else type(cur).__name__
            fatal(f"{label}: path {path!r} missing at {k!r}. Available here: {here}")
        cur = cur[k]
    return cur


def expected_edge_count(integrity: dict) -> int:
    """a2_integrity.json stores edges_by_label_relation as a list of {label, relation, len}."""
    rows = dig(integrity, ["edges_by_label_relation"], "a2_integrity")
    for r in rows:
        if r.get("label") == LABEL and r.get("relation") == RELATION:
            return int(r.get("len") or r.get("count"))
    fatal(f"{LABEL}/{RELATION} not found in edges_by_label_relation")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", default="data/gold/repodb_resolved.parquet")
    ap.add_argument("--edges-glob", default="data/canon/canon_edges/*.parquet")
    ap.add_argument("--a2-integrity", default="data/manifest/a2_integrity.json")
    ap.add_argument("--out", default="data/manifest/a5_circularity.json")
    args = ap.parse_args()

    integrity = json.loads(Path(args.a2_integrity).read_text())
    exp_edges = expected_edge_count(integrity)

    # 1. DRG-DIS/INDICATION edges, orientation-insensitive key. -----------------
    edges = (pl.scan_parquet(args.edges_glob)
               .filter((pl.col("label") == LABEL) & (pl.col("relation") == RELATION))
               .select(["from", "to", "sources_direct"])
               .collect())
    if edges.height != exp_edges:
        fatal(f"{LABEL}/{RELATION} edge count {edges.height:,} != a2_integrity {exp_edges:,}")
    edges = edges.with_columns(
        pl.min_horizontal("from", "to").alias("lo"),
        pl.max_horizontal("from", "to").alias("hi"),
    )

    # 2. Resolved gold POSITIVES that map to a real OKG pair. -------------------
    resolved = pl.read_parquet(args.resolved)
    pos = (resolved
           .filter((pl.col("resolved_class") == "positive")
                   & pl.col("okg_drug_id").is_not_null()
                   & pl.col("okg_dis_id").is_not_null())
           .with_columns(
               pl.min_horizontal("okg_drug_id", "okg_dis_id").alias("lo"),
               pl.max_horizontal("okg_drug_id", "okg_dis_id").alias("hi"))
           .select(["lo", "hi"]).unique())
    n_eval = pos.height

    # 3. Overlap. ---------------------------------------------------------------
    matched = pos.join(edges.select(["lo", "hi", "sources_direct"]), on=["lo", "hi"], how="inner")
    n_matched = matched.select(["lo", "hi"]).unique().height
    frac = (n_matched / n_eval) if n_eval else 0.0

    # 4. Source share on the matched edges — expect DrugCentral to dominate. ----
    src = (matched.select("sources_direct").explode("sources_direct")
                  .drop_nulls("sources_direct")
                  .group_by("sources_direct").len().sort("len", descending=True))
    src_share = {r["sources_direct"]: int(r["len"]) for r in src.iter_rows(named=True)}

    report = {
        "stage": "A5-circularity",
        "drg_dis_indication_edges": edges.height,
        "evaluable_positive_okg_pairs": n_eval,
        "already_an_indication_edge": n_matched,
        "fraction_already_edge": round(frac, 4),
        "sources_direct_on_matched_edges": src_share,
        "interpretation": (
            "repoDB is DrugCentral-built and DRG-DIS carries DrugCentral IDs; a high fraction here "
            "means the gold positives and the graph's indication edges are largely the same upstream "
            "data. A7 removes these edges, so the design holds, but the coverage number is inflated "
            "and any 'novel' recovery of such a pair is not independent. STATE THIS IN THE WRITEUP."
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"circularity: {n_matched:,}/{n_eval:,} evaluable positives "
          f"({frac:.1%}) already a {LABEL}/{RELATION} edge")
    print(f"sources_direct on matched: {src_share}")
    print(f"[A5] wrote {args.out}")


if __name__ == "__main__":
    main()
