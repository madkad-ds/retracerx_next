#!/usr/bin/env python3
"""
ReTraceRx Phase B — B3.1: discover the REAL graph vocabulary.

The how-to's metapath uses placeholder relation names ("AFFECTS", ...). Your graph
stores COMPOSITE relationship TYPES (LABEL__RELATION, 50 surviving after leakage
filtering). This script lists what's actually there so you build the metapath from
real names. Also verifies you're on the leakage-filtered variant and APOC is present.

Writes: data/retrieval/b3_rel_types.json  (types, bare relations, node labels)
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path

import polars as pl
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PWD = os.environ.get("NEO4J_PASSWORD", "retracerx")
DB = os.environ.get("NEO4J_DB", "neo4j")
A3_TYPE_TABLE = Path("data/a3_type_table.parquet")  # optional: census counts
OUT = Path("data/retrieval/b3_rel_types.json")


def main():
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    with drv.session(database=DB) as s:
        # APOC preflight — B3 retrieval needs apoc.path.expandConfig
        try:
            v = s.run("RETURN apoc.version() AS v").single()["v"]
            apoc = f"APOC {v}"
        except Exception as e:
            apoc = f"MISSING ({e})"

        # leakage-variant marker (A3 Step 5 writes a :_Meta singleton)
        try:
            meta = s.run("MATCH (m:_Meta) RETURN properties(m) AS p").single()
            meta = meta["p"] if meta else None
        except Exception:
            meta = None

        rel_types = sorted(s.run(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType AS t").value())
        node_labels = sorted(s.run(
            "CALL db.labels() YIELD label RETURN label").value())
    drv.close()

    # split composite LABEL__RELATION -> (label, relation)
    by_relation = defaultdict(list)
    for t in rel_types:
        rel = t.split("__", 1)[1] if "__" in t else t
        by_relation[rel].append(t)

    counts = {}
    if A3_TYPE_TABLE.exists():
        tt = pl.read_parquet(A3_TYPE_TABLE)
        # column name for composite type varies; find it heuristically
        ccol = next((c for c in tt.columns if "type" in c.lower()), None)
        ncol = next((c for c in tt.columns if "count" in c.lower() or "census" in c.lower()), None)
        if ccol and ncol:
            counts = {r[ccol]: r[ncol] for r in tt.select([ccol, ncol]).iter_rows(named=True)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "apoc": apoc,
        "meta_marker": meta,
        "n_relationship_types": len(rel_types),
        "relationship_types": rel_types,
        "bare_relations": {r: ts for r, ts in sorted(by_relation.items())},
        "node_labels": node_labels,
        "census_counts": counts,
    }, indent=2, default=str))  # default=str: Neo4j DateTime in _Meta isn't JSON-native

    print(f"{apoc}")
    print(f"variant marker: {meta}")
    print(f"relationship types: {len(rel_types)} (expect 50)")
    print(f"node labels: {node_labels}")
    print("\nBARE RELATIONS present (use these to define the metapath):")
    for r, ts in sorted(by_relation.items()):
        tot = sum(counts.get(t, 0) for t in ts) if counts else None
        print(f"  {r:35s} x{len(ts):2d} composite types"
              + (f"  (~{tot:,} edges)" if tot else ""))
    print(f"\nwrote {OUT}")
    if "MISSING" in apoc:
        print("\n!!! APOC missing — B3 retrieval needs it. Recreate the container with:")
        print("    -e NEO4J_PLUGINS='[\"apoc\"]'   (data volume persists; password NOT reset)")


if __name__ == "__main__":
    main()
