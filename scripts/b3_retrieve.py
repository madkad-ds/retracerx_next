#!/usr/bin/env python3
"""
ReTraceRx Phase B — B3.3: metapath-constrained candidate retrieval.

Given a disease graph_id (verbatim id from B2's link_entity), return raw evidence
paths that traverse mechanism (never a direct drug-disease edge — that label is gone)
and terminate on a REACHABLE whitelisted drug. Feeds B4 (rank) and B5 (serialize).

Corrections vs the how-to's B3 Cypher:
  * match on `id` (not id_curie); value = verbatim graph_id
  * relationshipFilter uses COMPOSITE types (LABEL__RELATION), discovered not guessed
  * labelFilter "/DRG" (node label code), not "/Drug"
  * whitelist = candidate_whitelist_reachable.csv (1,565), not the raw 1,700 (D35)
  * hub blacklist from b3_compute_hubs.py; transaction TIMEOUT to prevent hangs
  * undirected traversal (no >/< markers): the 13 undirected labels are stored
    single-orientation (D16), so directed traversal would miss half. Per-predicate
    direction is a documented refinement, not MVP.

Config (optional): data/retrieval/b3_metapath.json
    {"mech_relations": [...], "min_level": 2, "max_level": 3, "limit": 500, "timeout_s": 30}
If absent, MECH_RELATIONS below is used — VALIDATE these against b3_rel_types.json;
if none of them exist in the graph the script errors out (won't silently return nothing).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import polars as pl
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PWD = os.environ.get("NEO4J_PASSWORD", "retracerx")
DB = os.environ.get("NEO4J_DB", "neo4j")

WHITELIST = Path("data/holdout/candidate_whitelist_reachable.csv")  # 1,565 (D35)
HUBS = Path("data/retrieval/hub_nodes.parquet")
REL_TYPES = Path("data/retrieval/b3_rel_types.json")                # from B3.1
CONFIG = Path("data/retrieval/b3_metapath.json")                    # optional override

# Placeholder mechanistic relations from the how-to — VALIDATE against b3_rel_types.json.
MECH_RELATIONS = ["AFFECTS", "INTERACTS_WITH", "PARTICIPATES_IN",
                  "GENE_ASSOCIATED_WITH_CONDITION"]
MIN_LEVEL, MAX_LEVEL, LIMIT, TIMEOUT_S = 2, 3, 500, 30

_STATE: dict = {}


def _cfg():
    cfg = {"mech_relations": MECH_RELATIONS, "min_level": MIN_LEVEL,
           "max_level": MAX_LEVEL, "limit": LIMIT, "timeout_s": TIMEOUT_S}
    if CONFIG.exists():
        cfg.update(json.loads(CONFIG.read_text()))
    return cfg


def _load():
    if _STATE:
        return _STATE
    if not WHITELIST.exists():
        raise FileNotFoundError(f"{WHITELIST} missing (A8/D35 reachable whitelist).")
    if not REL_TYPES.exists():
        raise FileNotFoundError(f"{REL_TYPES} missing — run b3_discover_types.py first.")
    wl = pl.read_csv(WHITELIST)
    wcol = "okg_drug_id" if "okg_drug_id" in wl.columns else wl.columns[0]
    whitelist = wl[wcol].to_list()

    hub_ids = []
    if HUBS.exists():
        hub_ids = pl.read_parquet(HUBS)["id"].to_list()

    rt = json.loads(REL_TYPES.read_text())
    all_types = set(rt["relationship_types"])
    cfg = _cfg()
    # Prefer an explicit composite-type whitelist (exact endpoint control); fall back
    # to expanding bare mech_relations. Either way, validate against the real graph.
    if cfg.get("composite_types"):
        chosen = [t for t in cfg["composite_types"] if t in all_types]
        missing = sorted(set(cfg["composite_types"]) - all_types)
        if missing:
            raise RuntimeError(f"composite_types not in graph: {missing} — check {REL_TYPES}")
    else:
        chosen = [t for t in all_types if t.split("__", 1)[-1] in set(cfg["mech_relations"])]
    if not chosen:
        raise RuntimeError(
            "Empty metapath. Set composite_types (preferred) or mech_relations in "
            f"{CONFIG} to names present in {REL_TYPES} — refusing to run an empty metapath.")
    rel_filter = "|".join(sorted(chosen))  # no >/< => both directions (D16)

    _STATE.update(whitelist=set(whitelist), hub_ids=hub_ids,
                  rel_filter=rel_filter, cfg=cfg, n_types=len(chosen))
    return _STATE


CYPHER = """
MATCH (d:Node {id: $disease_id})
OPTIONAL MATCH (h:Node) WHERE h.id IN $hub_ids
WITH d, collect(h) AS hubs
CALL apoc.path.expandConfig(d, {
    relationshipFilter: $rel_filter,
    labelFilter: '/DRG',
    minLevel: $min_level, maxLevel: $max_level,
    bfs: true, limit: $limit,
    uniqueness: 'NODE_PATH',
    blacklistNodes: hubs
}) YIELD path
WITH path, last(nodes(path)) AS drug
WHERE drug.id IN $whitelist
RETURN drug.id AS drug_id, drug.name AS drug_name,
       [n IN nodes(path) | n.id]    AS node_ids,
       [n IN nodes(path) | n.label] AS node_labels,
       [n IN nodes(path) | n.name]  AS node_names,
       [r IN relationships(path) | type(r)]        AS rel_types,
       [r IN relationships(path) | r.relation]     AS rel_relations,
       [r IN relationships(path) | startNode(r).id] AS rel_starts,
       length(path) AS n_hops
LIMIT $limit
"""


def retrieve_paths(disease_id: str) -> list[dict]:
    """Return [{drug_graph_id, drug_name, paths:[{...raw path...}]}] for one disease."""
    S = _load()
    cfg = S["cfg"]
    params = dict(disease_id=disease_id, hub_ids=S["hub_ids"],
                  rel_filter=S["rel_filter"], whitelist=list(S["whitelist"]),
                  min_level=cfg["min_level"], max_level=cfg["max_level"],
                  limit=cfg["limit"])
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    try:
        with drv.session(database=DB) as s:
            with s.begin_transaction(timeout=cfg["timeout_s"]) as tx:
                rows = [r.data() for r in tx.run(CYPHER, **params)]
                tx.commit()
    finally:
        drv.close()

    by_drug: dict[str, dict] = {}
    for r in rows:
        g = by_drug.setdefault(r["drug_id"], {
            "drug_graph_id": r["drug_id"], "drug_name": r["drug_name"], "paths": []})
        g["paths"].append({
            "node_ids": r["node_ids"], "node_labels": r["node_labels"],
            "node_names": r["node_names"], "rel_types": r["rel_types"],
            "rel_relations": r["rel_relations"], "rel_starts": r["rel_starts"],
            "n_hops": r["n_hops"],
        })
    return list(by_drug.values())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("disease_id", help="verbatim graph_id, e.g. MONDO_0005148")
    ap.add_argument("--show", type=int, default=3, help="sample N drugs")
    a = ap.parse_args()
    S = _load()
    print(f"metapath: {S['n_types']} composite types | hubs blacklisted: {len(S['hub_ids'])} "
          f"| whitelist: {len(S['whitelist'])}")
    cands = retrieve_paths(a.disease_id)
    print(f"disease {a.disease_id} -> {len(cands)} candidate drugs "
          f"({sum(len(c['paths']) for c in cands)} paths)")
    for c in sorted(cands, key=lambda c: len(c["paths"]), reverse=True)[:a.show]:
        print(f"\n  {c['drug_graph_id']} ({c['drug_name']}) — {len(c['paths'])} paths")
        for p in c["paths"][:2]:
            chain = " -> ".join(
                f"{p['node_names'][i] or p['node_ids'][i]} =[{p['rel_relations'][i]}]="
                for i in range(len(p["rel_relations"]))) + f" {p['node_names'][-1] or p['node_ids'][-1]}"
            print(f"    ({p['n_hops']}h) {chain}")
