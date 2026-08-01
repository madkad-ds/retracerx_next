#!/usr/bin/env python3
"""
ReTraceRx Phase B — B3.2: precompute hub nodes to blacklist during traversal.

A hub (very high degree) explodes path-finding and manufactures spurious links.
We rank nodes by total degree (COUNT{(n)--()}; Neo4j 5 — size((n)--()) was removed)
and freeze a blacklist. GUARDS: never blacklist DRG (a candidate answer) or DIS
(the query) nodes — only intermediates (GEN/PWY/BPO/PHE/...).

Pick the cutoff from the printed distribution. Default: degree >= DEGREE_THRESHOLD.

Writes: data/retrieval/hub_nodes.parquet  (id, label, degree)
"""
from __future__ import annotations
import os
from pathlib import Path

import polars as pl
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PWD = os.environ.get("NEO4J_PASSWORD", "retracerx")
DB = os.environ.get("NEO4J_DB", "neo4j")

DEGREE_THRESHOLD = 3000          # blacklist intermediates with degree >= this
NEVER_BLACKLIST_LABELS = ("DRG", "DIS")
OUT = Path("data/retrieval/hub_nodes.parquet")


def main():
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    # Total degree per node. Store-backed COUNT{} is cheap per node.
    q = """
    MATCH (n:Node)
    WITH n, COUNT { (n)--() } AS degree
    WHERE degree >= $thr AND NOT n.label IN $never
    RETURN n.id AS id, n.label AS label, degree
    ORDER BY degree DESC
    """
    with drv.session(database=DB) as s:
        rows = s.run(q, thr=DEGREE_THRESHOLD, never=list(NEVER_BLACKLIST_LABELS)).data()
        # distribution snapshot (top intermediates), for choosing the cutoff
        dist = s.run("""
            MATCH (n:Node) WHERE NOT n.label IN $never
            WITH n.label AS label, COUNT { (n)--() } AS d
            RETURN label,
                   max(d) AS max_deg,
                   percentileDisc(d, 0.999) AS p999,
                   percentileDisc(d, 0.99) AS p99
            ORDER BY max_deg DESC
        """, never=list(NEVER_BLACKLIST_LABELS)).data()
    drv.close()

    df = pl.DataFrame(rows) if rows else pl.DataFrame(
        {"id": [], "label": [], "degree": []},
        schema={"id": pl.Utf8, "label": pl.Utf8, "degree": pl.Int64})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)

    print(f"degree threshold: {DEGREE_THRESHOLD}  (excludes {NEVER_BLACKLIST_LABELS})")
    print("per-label degree distribution (intermediates only):")
    for r in dist:
        print(f"  {r['label']:5s} max={r['max_deg']:>8}  p99.9={r['p999']:>7}  p99={r['p99']:>6}")
    print(f"\nblacklisted hubs (degree >= {DEGREE_THRESHOLD}): {df.height}")
    if df.height:
        print(df.group_by("label").len().sort("len", descending=True))
        print("top 10:")
        print(df.head(10))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
