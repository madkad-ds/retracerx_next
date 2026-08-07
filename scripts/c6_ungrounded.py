#!/usr/bin/env python3
"""
c6_ungrounded.py — the ungrounded-LLM baseline (plan step 9 / how-to C6).

Same Qwen reasoner, NO graph: prompt for a repurposing shortlist for the disease,
map the proposed drug NAMES to okg ids via the reachable whitelist, intersect with
that whitelist. Its picks carry NO retrieved paths, so when run through the SAME
verifier (c3_verify) every pick is ungrounded → hallucination ≈ 1.0 by
construction. The grounded-vs-ungrounded gap is the project's core positive result.

Only NAMES are generated under guided decoding (short, bounded array) — no free-text
mechanism field, because xgrammar doesn't cap string length (the C2 lesson). The
"claim" each pick makes ("this drug treats the disease") is implicit and, having no
path_id, counts as a hallucinated/ungrounded claim.
"""
from __future__ import annotations
import os
import re
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c0_config as C  # noqa: E402
from c1_reasoner import get_reasoner  # noqa: E402

_NAME2ID: dict | None = None
_ID2DISEASE: dict | None = None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _whitelist_name_map() -> dict:
    """Normalized drug name -> okg_drug_id, from the reachable whitelist
    (cols include name + okg_drug_id)."""
    global _NAME2ID
    if _NAME2ID is None:
        df = pl.read_csv(C.WHITELIST_REACHABLE)
        idc = "okg_drug_id" if "okg_drug_id" in df.columns else df.columns[0]
        namec = "name" if "name" in df.columns else None
        _NAME2ID = {}
        if namec:
            for r in df.select([idc, namec]).iter_rows(named=True):
                if r[namec]:
                    _NAME2ID.setdefault(_norm(r[namec]), r[idc])
        # pert_iname is another usable synonym column in this whitelist
        if "pert_iname" in df.columns:
            for r in df.select([idc, "pert_iname"]).iter_rows(named=True):
                if r["pert_iname"]:
                    _NAME2ID.setdefault(_norm(r["pert_iname"]), r[idc])
    return _NAME2ID


def _disease_name(disease_graph_id: str) -> str:
    """Resolve a DIS graph_id -> human name from canon_nodes (for the prompt)."""
    global _ID2DISEASE
    if _ID2DISEASE is None:
        _ID2DISEASE = {}
        if C.CANON_NODES.is_file():
            df = pl.read_parquet(C.CANON_NODES)
            if {"id", "name"}.issubset(df.columns):
                sub = df.filter(pl.col("label") == "DIS") if "label" in df.columns else df
                for r in sub.select(["id", "name"]).iter_rows(named=True):
                    if r["name"]:
                        _ID2DISEASE[str(r["id"])] = r["name"]
    return _ID2DISEASE.get(disease_graph_id, disease_graph_id)


def _map_names_to_whitelist(names, n2i):
    """Exact then fuzzy (rapidfuzz) name -> whitelist id; keep first hit, deduped."""
    from rapidfuzz import process, fuzz
    choices = list(n2i.keys())
    out, seen = [], set()
    for nm in names:
        q = _norm(nm)
        did = n2i.get(q)
        if did is None and choices:
            m = process.extractOne(q, choices, scorer=fuzz.WRatio, score_cutoff=90)
            if m:
                did = n2i[m[0]]
        if did and did not in seen:
            out.append({"drug_graph_id": did, "proposed_name": nm})
            seen.add(did)
    return out


def run_ungrounded(disease_graph_id: str, n: int = 10) -> dict:
    """Return {picks: [{drug_graph_id, proposed_name, cited_path_ids: []}], ...}.
    cited_path_ids is empty by construction — the baseline saw no graph."""
    disease = _disease_name(disease_graph_id)
    schema = {
        "type": "object",
        "properties": {"drugs": {"type": "array", "maxItems": n,
                                 "items": {"type": "string"}}},
        "required": ["drugs"],
    }
    prompt = (
        f"List up to {n} existing drugs you would consider repurposing to treat "
        f"{disease}. Give generic drug names only, most promising first. Reply as JSON."
    )
    out = get_reasoner().generate_json(prompt, schema, max_new_tokens=384)
    names = out.get("drugs", [])
    mapped = _map_names_to_whitelist(names, _whitelist_name_map())
    for m in mapped:
        m["cited_path_ids"] = []                 # ungrounded: no retrieved paths
    return {"disease_graph_id": disease_graph_id, "disease_name": disease,
            "proposed_names": names, "picks": mapped,
            "ranked_drug_ids": [m["drug_graph_id"] for m in mapped],
            "n_proposed": len(names), "n_mapped": len(mapped)}


import copy         # noqa: E402
import inspect      # noqa: E402
import b6_tools as b6  # noqa: E402


def _retrieve_pool(disease_graph_id, split, mode):
    """Same retrieval the agent uses; returns deep-copied contract candidates."""
    params = inspect.signature(b6.retrieve_candidates).parameters
    kw = {}
    if "split" in params:
        kw["split"] = split
    if "mode" in params:
        kw["mode"] = mode
    contract = b6.retrieve_candidates(disease_graph_id, **kw)
    cands = (contract["candidates"] if isinstance(contract, dict) and "candidates" in contract
             else contract)
    return copy.deepcopy(cands)                     # caches READ-ONLY (D-B21)


def run_ungrounded_pool(disease_graph_id, split="disease_cold", mode="plus_true",
                        top_k=None):
    """FAIR MIDDLE BASELINE: give the LLM the retrieved candidate DRUGS (names only,
    no paths) and let it rank among them. Same pool as the grounded agent, so
    recovery is on equal footing; still no paths to cite, so hallucination stays
    ~1.0 by construction. Isolates 'does grounding help RANK the true drug' from
    'who saw more candidates'."""
    disease = _disease_name(disease_graph_id)
    cands = _retrieve_pool(disease_graph_id, split, mode)
    if top_k:
        cands = cands[:top_k]
    pool = [(c["drug_graph_id"], c.get("drug_name") or c["drug_graph_id"]) for c in cands]
    if not pool:
        return {"disease_graph_id": disease_graph_id, "disease_name": disease,
                "picks": [], "ranked_drug_ids": [], "pool_ids": [], "n_pool": 0}
    pool_ids = [pid for pid, _ in pool]
    listing = "\n".join(f"  {nm} ({pid})" for pid, nm in pool)
    schema = {"type": "object",
              "properties": {"ranked_drug_ids": {"type": "array",
                                                 "items": {"type": "string", "enum": pool_ids}}},
              "required": ["ranked_drug_ids"]}
    prompt = (f"Candidate drugs for repurposing to treat {disease}:\n{listing}\n\n"
              "Rank these drug ids from most to least promising for this disease. "
              "Use ONLY ids from the list above. Reply as JSON.")
    out = get_reasoner().generate_json(prompt, schema, max_new_tokens=512)
    ranked = [d for d in out.get("ranked_drug_ids", []) if d in set(pool_ids)]
    seen = set(ranked)
    ranked += [d for d in pool_ids if d not in seen]     # keep pool coverage for recall parity
    picks = [{"drug_graph_id": d, "cited_path_ids": []} for d in ranked]  # ungrounded: no paths
    return {"disease_graph_id": disease_graph_id, "disease_name": disease,
            "picks": picks, "ranked_drug_ids": ranked,
            "pool_ids": pool_ids, "n_pool": len(pool_ids)}


class UngroundedRanker:
    """Harness adapter (C5): rank(disease, universe) -> list[str]."""
    mode_aware = False

    def __init__(self, n: int = 10):
        self.n = n

    def rank(self, disease: str, universe: set[str]) -> list[str]:
        picks = run_ungrounded(disease, self.n)["picks"]
        ranked = [p["drug_graph_id"] for p in picks if p["drug_graph_id"] in universe]
        seen = set(ranked)
        ranked += [d for d in universe if d not in seen]
        return ranked


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease-id", default="MONDO_0005148")
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    print(json.dumps(run_ungrounded(a.disease_id, a.n), indent=2))
