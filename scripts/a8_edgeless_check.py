#!/usr/bin/env python3
"""
A8 pre-flight diagnostic: which candidate/holdout drugs are edgeless in the FILTERED graph?

A drug with no edges after leakage removal cannot be reached by any path and cannot be
embedded by KGE -> it is unscoreable. If a holdout-POSITIVE drug is edgeless, it is an
automatic miss (report as a coverage boundary, not a model failure).

Run from repo root:  python scripts/a8_edgeless_check.py
"""
import sys
import polars as pl

CANON_NODES = "data/canon/canon_nodes.parquet"
CANON_EDGES = "data/canon/canon_edges/*.parquet"
REMOVED     = "data/holdout/removed_edge_ids.csv"
WHITELIST   = "data/canon/candidate_whitelist.parquet"
HOLDOUT_POS = "data/holdout/holdout_positives.csv"
EXPECT_EDGELESS = 4_333          # M-A3-1 cross-check

# --- recompute the filtered edge set: canon_edges ANTI removed(endpoint_key, relation) ---
removed = pl.read_csv(REMOVED).select(["endpoint_key", "relation"]).unique().lazy()
filtered = (pl.scan_parquet(CANON_EDGES)
              .join(removed, on=["endpoint_key", "relation"], how="anti"))

# every node id that still appears on at least one edge
connected = (pl.concat([filtered.select(pl.col("from").alias("id")),
                        filtered.select(pl.col("to").alias("id"))])
               .unique().collect().to_series())

# DRG nodes with zero edges in this variant
drg = pl.read_parquet(CANON_NODES).filter(pl.col("label") == "DRG").select("id").to_series()
edgeless = set(drg.to_list()) - set(connected.to_list())
print(f"edgeless DRG nodes in filtered variant: {len(edgeless):,} (M-A3-1 expected {EXPECT_EDGELESS:,})")
if len(edgeless) != EXPECT_EDGELESS:
    print(f"  [WARN] does not match M-A3-1 {EXPECT_EDGELESS:,} - investigate before trusting.",
          file=sys.stderr)

# --- intersect with the two populations ---
def report(name, ids):
    ids = set(ids)
    hit = ids & edgeless
    pct = 100 * len(hit) / len(ids) if ids else 0
    print(f"{name}: {len(hit):,} / {len(ids):,} ({pct:.1f}%) are edgeless in the filtered graph")
    return hit

wl = pl.read_parquet(WHITELIST)["okg_drug_id"].to_list()
hp = pl.read_csv(HOLDOUT_POS)["okg_drug_id"].to_list()

print()
wl_hit = report("candidate whitelist (1,700)", wl)
hp_hit = report("holdout-positive drugs      ", hp)

# holdout-positive drugs that are edgeless are unavoidable misses -> write them out
if hp_hit:
    (pl.read_csv(HOLDOUT_POS)
       .filter(pl.col("okg_drug_id").is_in(list(hp_hit)))
       .write_csv("data/holdout/holdout_positives_unscoreable_drugs.csv"))
    print(f"\n  -> wrote {len(hp_hit)} unavoidable-miss holdout rows to "
          f"data/holdout/holdout_positives_unscoreable_drugs.csv (report as coverage boundary)")
