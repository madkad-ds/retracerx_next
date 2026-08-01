#!/usr/bin/env python3
"""
ReTraceRx Phase B — B5: serialize paths to evidence + assemble the v1.0 contract.

Turns each B4-kept path into the frozen retrieval-contract per-path object:
    {path_id, triples, nl_evidence, sources}
and assembles the full contract (query_disease from B2 + ranked candidates from B4).

Design facts:
  * triples are in PATH order (disease-first: node_ids[0]=disease ... node_ids[-1]=drug),
    RELATION = the COMPOSITE type (DRG_GEN__TARGET) per the frozen contract semantics.
  * path_id = compute_path_id(triples) — a canonical-JSON sha1. Phase C's faithfulness
    verifier MUST import THIS function so its derivation is identical. Single source of truth.
  * sources are NOT in Neo4j (D15). We join back to data/canon/canon_edges/<LABEL>.parquet
    (sources_direct), matching edges by unordered endpoints + bare relation.
  * verifier note: triples are path-oriented; existence checks must be UNDIRECTED (D16 —
    13 undirected labels are stored single-orientation, so a directed check misses half).

CLI:  python b5_serialize.py "type 2 diabetes" --split disease_cold --out contract.json
      python b5_serialize.py MONDO_0005148 --id --split disease_cold      # skip the linker
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

import polars as pl

import b2_link
import b4_rank

CANON_EDGES = Path("data/canon/canon_edges")


# ---------------------------------------------------------- path_id (SHARED)
def compute_path_id(triples: list[list[str]]) -> str:
    """Deterministic, language-stable path id. Phase C's verifier imports THIS.
    Recipe: sha1 of compact canonical JSON of the ordered triples, first 10 hex."""
    canon = json.dumps(triples, separators=(",", ":"), ensure_ascii=False)
    return "p_" + hashlib.sha1(canon.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------- helpers
def split_composite(ctype: str) -> tuple[str, str]:
    """DRG_GEN__TARGET -> ('DRG-GEN', 'TARGET'). Label underscore came from a hyphen
    (Neo4j type constraint); canon_edges/filenames use the hyphen form."""
    lbl, rel = ctype.split("__", 1)
    return lbl.replace("_", "-"), rel


PHRASE = {
    "ASSOCIATED_WITH": "is associated with", "INTERACTS_WITH": "interacts with",
    "PHENOTYPE_PRESENT": "presents phenotype", "TARGET": "is a target of",
    "INHIBITOR": "is inhibited by", "AGONIST": "is agonised by",
    "ANTAGONIST": "is antagonised by", "ACTIVATOR": "is activated by",
    "MODULATOR": "is modulated by", "POSITIVE_MODULATOR": "is positively modulated by",
    "NEGATIVE_MODULATOR": "is negatively modulated by",
    "POSITIVE_ALLOSTERIC_MODULATOR": "is positively allosterically modulated by",
    "NEGATIVE_ALLOSTERIC_MODULATOR": "is negatively allosterically modulated by",
    "ALLOSTERIC_ANTAGONIST": "is allosterically antagonised by",
    "PARTIAL_AGONIST": "is partially agonised by", "INVERSE_AGONIST": "is inverse-agonised by",
    "BLOCKER": "is blocked by", "OPENER": "is opened by", "BINDING_AGENT": "is bound by",
    "RELEASING_AGENT": "is released by", "STABILISER": "is stabilised by",
    "DEGRADER": "is degraded by", "CARRIER": "is carried by",
    "TRANSPORTER": "is transported by", "ENZYME": "is enzymatically processed by",
    "SUBSTRATE": "is a substrate of",
}


def _phrase(rel: str) -> str:
    return PHRASE.get(rel, "is linked (%s) to" % rel.lower().replace("_", " "))


def to_curie(graph_id: str) -> str:
    """Display CURIE for a drug id. CHEMBL/DrugBank prettified; else verbatim."""
    m = re.match(r"^(CHEMBL)(\d+)$", graph_id)
    if m:
        return f"CHEMBL:{m.group(2)}"
    if re.match(r"^DB\d+$", graph_id):
        return f"DrugBank:{graph_id}"
    return graph_id


# ---------------------------------------------------------- sources join
def build_sources_map(all_triples: list[tuple[str, str, str]]) -> dict:
    """{(frozenset{head,tail}, bare_relation): [sources]} from canon_edges.
    Loads only the label files actually referenced; filters by endpoint membership."""
    nodes, labels = set(), set()
    for h, ct, t in all_triples:
        lbl, _ = split_composite(ct)
        labels.add(lbl); nodes.add(h); nodes.add(t)
    node_list = list(nodes)
    smap: dict = {}
    for lbl in labels:
        f = CANON_EDGES / f"{lbl}.parquet"
        if not f.exists():
            continue
        df = (pl.scan_parquet(f)
                .filter(pl.col("from").is_in(node_list) & pl.col("to").is_in(node_list))
                .select(["from", "to", "relation", "sources_direct"])
                .collect())
        for r in df.iter_rows(named=True):
            key = (frozenset({r["from"], r["to"]}), r["relation"])
            src = r["sources_direct"] or []
            smap[key] = sorted(set(src) | set(smap.get(key, [])))
    return smap


def serialize_path(path: dict, smap: dict) -> dict:
    ids, ctypes, names = path["node_ids"], path["rel_types"], path["node_names"]
    triples = [[ids[i], ctypes[i], ids[i + 1]] for i in range(len(ctypes))]
    parts, srcs = [], set()
    for i, (h, ct, t) in enumerate(triples):
        _, rel = split_composite(ct)
        hn = names[i] or ids[i]
        tn = names[i + 1] or ids[i + 1]
        parts.append(f"{hn} {_phrase(rel)} {tn}")
        srcs |= set(smap.get((frozenset({h, t}), rel), []))
    return {"path_id": compute_path_id(triples), "triples": triples,
            "nl_evidence": "; ".join(parts) + ".", "sources": sorted(srcs)}


# ---------------------------------------------------------- contract assembly
def build_contract(query: str, is_id: bool = False, split: str | None = None,
                   k: int | None = None, m: int | None = None) -> dict:
    if is_id:
        qd = {"text": query, "graph_id": query, "curie": query,
              "link_score": 1.0, "alternatives": []}
        disease_id = query
    else:
        qd = b2_link.link_entity(query)
        disease_id = qd.get("graph_id")

    contract = {"contract_version": "1.0", "query_disease": qd, "candidates": []}
    if not disease_id:  # unlinked / low confidence — agent disambiguates via alternatives
        return contract

    ranked = b4_rank.rank_candidates(disease_id, split=split, k=k, m=m)
    all_triples = [(p["node_ids"][i], p["rel_types"][i], p["node_ids"][i + 1])
                   for c in ranked for p in c["paths"]
                   for i in range(len(p["rel_types"]))]
    smap = build_sources_map(all_triples)

    for c in ranked:
        contract["candidates"].append({
            "rank": c["rank"],
            "drug_graph_id": c["drug_graph_id"],
            "drug_curie": to_curie(c["drug_graph_id"]),
            "drug_name": c["drug_name"],
            "score": round(c["score"], 6),
            "kge_score": (round(c["kge_score"], 6) if c["kge_score"] is not None else None),
            "support_count": c["support_count"],
            "specificity": round(c["specificity"], 6),
            "paths": [serialize_path(p, smap) for p in c["paths"]],
        })
    return contract


def _assert_shape(contract: dict):
    assert contract["contract_version"] == "1.0"
    for key in ("text", "graph_id", "curie", "link_score", "alternatives"):
        assert key in contract["query_disease"], f"query_disease missing {key}"
    for c in contract["candidates"]:
        for key in ("rank", "drug_graph_id", "drug_curie", "drug_name", "score",
                    "kge_score", "support_count", "specificity", "paths"):
            assert key in c, f"candidate missing {key}"
        for p in c["paths"]:
            assert set(p) == {"path_id", "triples", "nl_evidence", "sources"}
            assert p["path_id"] == compute_path_id(p["triples"]), "path_id not reproducible"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="disease text, or a graph_id with --id")
    ap.add_argument("--id", action="store_true", help="treat query as a verbatim graph_id")
    ap.add_argument("--split", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--out", default=None, help="write full contract JSON here")
    a = ap.parse_args()

    contract = build_contract(a.query, is_id=a.id, split=a.split, k=a.k, m=a.m)
    _assert_shape(contract)  # fails loud if the shape or a path_id drifts

    qd = contract["query_disease"]
    print(f"query: {qd['text']} -> {qd['graph_id']} (link {qd['link_score']}, "
          f"{len(qd['alternatives'])} alternatives)")
    print(f"candidates: {len(contract['candidates'])}")
    for c in contract["candidates"][:5]:
        kge = f"{c['kge_score']:.3f}" if c["kge_score"] is not None else "null"
        print(f"  #{c['rank']:>2} {c['drug_name']} (score={c['score']:.4f}, kge={kge}, "
              f"paths={len(c['paths'])})")
        for p in c["paths"][:1]:
            print(f"      {p['path_id']}  sources={p['sources']}")
            print(f"      {p['nl_evidence']}")
    if a.out:
        Path(a.out).write_text(json.dumps(contract, indent=2, ensure_ascii=False))
        print(f"\nwrote {a.out}")
