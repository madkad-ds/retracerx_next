#!/usr/bin/env python3
"""A8 / D35 — DIS-reachability over the FILTERED graph.

With DRG-DIS gone (E3), a drug is scoreable only if it has a path to some DIS node
(drug -> gene/pathway/phenotype -> ... -> disease). The exact, seconds-fast test is:
a drug reaches some disease IFF its weakly-connected component contains a DIS node.

Uses GDS WCC (Community edition, unlicensed — WCC is free). Do NOT use per-drug bounded
shortestPath: unreachable drugs force a full K-hop undirected blow-up through DIS-GEN(9.7M)
and ANA-GEN(8.8M) hubs and it was killed at >5 min (failure A8-1).

Preconditions:
  - retracerx-neo4j up, FILTERED variant live in neo4j/data (rel count == 21,733,867)
  - GDS plugin loaded (RETURN gds.version() works)

Guards:
  - unreachable MUST be a superset of the edgeless (degree-0) drugs   [hard assert]
  - unreachable SHOULD be strictly larger than edgeless               [warn if equal:
    that would mean WCC only recovered the edgeless floor and missed the
    "has edges but no path to any DIS" population, which is the whole point]

Outputs:
  data/holdout/a8_unreachable_drugs.parquet   graph-wide unreachable DRG + membership flags
  data/holdout/a8_reachability.json           counts per population / per split
  data/holdout/candidate_whitelist_reachable.csv   whitelist minus unreachable (hygiene)
"""
import json
import sys
from pathlib import Path

import polars as pl
from neo4j import GraphDatabase

# ---------------------------------------------------------------- config
URI, AUTH = "bolt://localhost:7687", ("neo4j", "retracerx")
GRAPH = "okg_filtered"

WHITELIST   = "data/canon/candidate_whitelist.parquet"      # 1,700 rows, col okg_drug_id
SPLIT_DIR   = Path("data/holdout/splits")                   # {split}_holdout_positives.csv
SPLITS      = ["disease_cold", "drug_cold", "random"]
# A8 pre-flight edgeless floor (disease-cold holdout positives whose drug is edgeless):
EDGELESS_PREFLIGHT = "data/holdout/holdout_positives_unscoreable_drugs.csv"  # 13 rows

OUT_PARQUET = Path("data/holdout/a8_unreachable_drugs.parquet")
OUT_JSON    = Path("data/holdout/a8_reachability.json")
OUT_WL_CLEAN = Path("data/holdout/candidate_whitelist_reachable.csv")

EXPECT_FILTERED_RELS = 21_733_867   # confirms the FILTERED store is attached, not the full one

# ---------------------------------------------------------------- Cypher
# WCC returns every DRG whose component holds no DIS node. Two streams: first collect
# the (small) set of component ids that contain a DIS, then stream again and keep the DRG
# nodes not in that set. Collecting component *ids* (ints), never nodes.
Q_UNREACHABLE = """
CALL gds.wcc.stream($graph) YIELD nodeId, componentId
WITH componentId, gds.util.asNode(nodeId) AS n
WHERE 'DIS' IN labels(n)
WITH collect(DISTINCT componentId) AS dis_components
CALL gds.wcc.stream($graph) YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS n, componentId, dis_components
WHERE 'DRG' IN labels(n) AND NOT componentId IN dis_components
RETURN n.id AS okg_drug_id
"""

Q_EDGELESS = "MATCH (d:DRG) WHERE NOT (d)--() RETURN d.id AS okg_drug_id"
Q_REL_COUNT = "MATCH ()-[r]->() RETURN count(r) AS c"
Q_GDS_VER = "RETURN gds.version() AS v"


def die(msg: str) -> None:
    sys.exit(f"[A8-reach FATAL] {msg}")


def main() -> None:
    drv = GraphDatabase.driver(URI, auth=AUTH)
    with drv.session() as s:
        # 0) sanity: GDS present + FILTERED store attached
        try:
            ver = s.run(Q_GDS_VER).single()["v"]
        except Exception as e:  # noqa: BLE001
            die(f"gds.version() failed — plugin not loaded ({e})")
        print(f"[A8-reach] GDS {ver}")

        rels = s.run(Q_REL_COUNT).single()["c"]
        if rels != EXPECT_FILTERED_RELS:
            die(f"relationship count {rels:,} != {EXPECT_FILTERED_RELS:,} — "
                f"the FILTERED variant is not the live store (wrong bind mount?)")
        print(f"[A8-reach] filtered store confirmed — {rels:,} relationships")

        # 1) (re)project the whole graph, UNDIRECTED
        s.run(f"CALL gds.graph.drop('{GRAPH}', false)")   # false => no error if absent
        proj = s.run(
            "CALL gds.graph.project($graph, '*', "
            "{ALL: {type: '*', orientation: 'UNDIRECTED'}}) "
            "YIELD nodeCount, relationshipCount",
            graph=GRAPH,
        ).single()
        print(f"[A8-reach] projected {proj['nodeCount']:,} nodes / "
              f"{proj['relationshipCount']:,} rels (undirected ~2x stored)")

        # 2) WCC -> unreachable DRG ids (graph-wide)
        unreachable = [r["okg_drug_id"] for r in s.run(Q_UNREACHABLE, graph=GRAPH)]
        # 3) edgeless (degree-0) DRG ids — the floor the assert checks against
        edgeless = [r["okg_drug_id"] for r in s.run(Q_EDGELESS)]

        s.run(f"CALL gds.graph.drop('{GRAPH}', false)")   # release heap

    drv.close()

    unreach_set, edgeless_set = set(unreachable), set(edgeless)
    print(f"[A8-reach] unreachable DRG (graph-wide) = {len(unreach_set):,}")
    print(f"[A8-reach] edgeless   DRG (degree 0)    = {len(edgeless_set):,}")

    # ---- GUARD 1 (hard): unreachable must be a superset of edgeless
    missing = edgeless_set - unreach_set
    if missing:
        die(f"{len(missing)} edgeless drugs are NOT in the unreachable set — "
            f"reachability query is wrong (orientation? wrong store?). e.g. {list(missing)[:5]}")
    print("[A8-reach] GUARD 1 ok — unreachable superset of edgeless")

    # ---- GUARD 2 (warn): unreachable should be strictly larger than edgeless
    if len(unreach_set) == len(edgeless_set):
        print("[A8-reach] WARN unreachable == edgeless — WCC recovered only the floor; "
              "the 'has edges but no path to DIS' population is missing. Investigate.")

    # cross-check the pre-flight floor if present (disease-cold holdout: 13)
    if Path(EDGELESS_PREFLIGHT).exists():
        pf = set(pl.read_csv(EDGELESS_PREFLIGHT)["okg_drug_id"])
        stray = pf - unreach_set
        if stray:
            die(f"{len(stray)} pre-flight edgeless holdout drugs missing from unreachable")
        print(f"[A8-reach] pre-flight floor ({len(pf)}) ⊆ unreachable ✓")

    # 4) membership flags: whitelist + each split's holdout positives
    wl = set(pl.read_parquet(WHITELIST)["okg_drug_id"])
    per_split: dict[str, list[str]] = {}
    for sp in SPLITS:
        fp = SPLIT_DIR / f"{sp}_holdout_positives.csv"
        if not fp.exists():
            print(f"[A8-reach] WARN split file missing, skipping: {fp}")
            continue
        drugs = set(pl.read_csv(fp)["okg_drug_id"])
        per_split[sp] = sorted(drugs & unreach_set)
        print(f"[A8-reach] {sp}: {len(per_split[sp])} / {len(drugs)} "
              f"holdout-positive drugs unreachable")

    wl_unreach = sorted(wl & unreach_set)
    print(f"[A8-reach] whitelist: {len(wl_unreach)} / {len(wl)} unreachable")

    # 5) freeze the graph-wide unreachable list with membership flags
    df = pl.DataFrame({"okg_drug_id": sorted(unreach_set)}).with_columns(
        in_whitelist=pl.col("okg_drug_id").is_in(list(wl)),
        is_edgeless=pl.col("okg_drug_id").is_in(list(edgeless_set)),
    )
    for sp, ids in per_split.items():
        df = df.with_columns(pl.col("okg_drug_id").is_in(ids).alias(f"holdout_{sp}"))
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PARQUET)

    # 6) whitelist hygiene — emit the reachable-only whitelist (declare-and-drop, D35)
    wl_full = pl.read_parquet(WHITELIST)
    wl_clean = wl_full.filter(~pl.col("okg_drug_id").is_in(wl_unreach))
    wl_clean.write_csv(OUT_WL_CLEAN)
    print(f"[A8-reach] clean whitelist: {wl_clean.height} / {wl_full.height} "
          f"(dropped {len(wl_unreach)} unreachable)")

    # 7) record counts
    OUT_JSON.write_text(json.dumps({
        "stage": "A8/D35",
        "method": "gds.wcc component-contains-DIS",
        "gds_version": ver,
        "unreachable_drugs_graphwide": len(unreach_set),
        "edgeless_drugs_degree0": len(edgeless_set),
        "whitelist_unreachable": len(wl_unreach),
        "whitelist_clean_rows": wl_clean.height,
        "holdout_unreachable_by_split": {k: len(v) for k, v in per_split.items()},
        "guard1_superset_of_edgeless": True,
        "guard2_strictly_larger": len(unreach_set) > len(edgeless_set),
        "whitelist_policy": "DROP unreachable from candidate_whitelist for eval (D35)",
    }, indent=2))
    print(f"[A8-reach] wrote {OUT_PARQUET}, {OUT_WL_CLEAN}, {OUT_JSON}")


if __name__ == "__main__":
    main()
