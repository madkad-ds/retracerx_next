#!/usr/bin/env python3
"""
c2_agent.py — Phase C core agent (LangGraph).

A shallow, deterministic pipeline (a fixed plan beats an open ReAct loop for a
scored eval, plan step 8):

    link -> retrieve -> reason -> verify -> assemble -> (ranking | abstention)

The agent CANNOT recover novel candidates absent from the 2-hop pool — that
ceiling is structural (Phase B C-1). Its job is to *justify*, *re-order by
faithful mechanistic support*, *demote unfaithful*, and *abstain* when nothing is
supported. That is the Phase C contribution, not recovery.

REASONING IS PER-CANDIDATE (token-safe): the reasoner scores one drug per call,
emitting a small flat JSON object. This never truncates (vs one big ranking array
that overran the budget and stopped before closing required fields), tightens each
drug's citation enum to its OWN paths, and keeps one bad generation from losing
the whole ranking. Deterministic (greedy + fixed seed).

Tools (Phase B, deterministic + cached + READ-ONLY, D-B21):
    entity_linker = b6_tools.link_entity
    kg_pathfinder = b6_tools.retrieve_candidates      (returns contract v1.0)
    kge_scorer    = already applied -> contract candidates carry `kge_score`
TxGemma property tool is Stretch; a no-op hook is left in reason_node.

Faithfulness: guided decoding constrains cited_path_ids to each candidate's path
enum (C4); verify re-derives each path_id via the SHARED compute_path_id
(never reimplemented; undirected handled inside it, D-B18/B19). The full C3
verifier replaces `_verify_entry`.

Install once (light):  uv pip install langgraph
"""
from __future__ import annotations
import copy
import inspect
import json
import os
import sys
from typing import Optional, TypedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c0_config as C  # noqa: E402
from b5_serialize import compute_path_id  # noqa: E402 — shared, never reimplemented (D-B18)
import b6_tools as b6  # noqa: E402
from c1_reasoner import get_reasoner  # noqa: E402


# =====================================================================
# State
# =====================================================================
class AgentState(TypedDict, total=False):
    disease_text: Optional[str]
    disease_graph_id: Optional[str]
    split: str
    mode: str
    query_disease: dict
    candidates: list           # contract candidates (DEEP-COPIED; caches read-only)
    path_index: dict           # {path_id: path_obj} across candidates, for verify
    reason_out: dict           # {"per_candidate": {drug_id: {...}}, ...}
    abstain: bool
    abstain_reason: Optional[str]
    result: dict


# =====================================================================
# Tool-call adapter (introspect real signatures; don't guess kwargs)
# =====================================================================
def _call_with_supported(fn, first, split, mode):
    params = inspect.signature(fn).parameters
    kw = {}
    if "split" in params:
        kw["split"] = split
    if "mode" in params:
        kw["mode"] = mode
    return fn(first, **kw)


# =====================================================================
# Nodes
# =====================================================================
def link_node(state: AgentState) -> dict:
    if state.get("disease_graph_id"):
        gid = state["disease_graph_id"]
        return {"query_disease": {"graph_id": gid, "text": gid, "link_score": 1.0,
                                  "alternatives": []}}
    text = state.get("disease_text")
    if not text:
        return {"abstain": True, "abstain_reason": "no disease_text or disease_graph_id given"}
    qd = b6.link_entity(text)
    if qd.get("graph_id"):
        return {"query_disease": qd, "disease_graph_id": qd["graph_id"]}
    alts = sorted(qd.get("alternatives", []),
                  key=lambda a: a.get("link_score", 0.0), reverse=True)
    if alts and alts[0].get("link_score", 0.0) >= C.LINK_ALT_MIN:
        return {"query_disease": qd, "disease_graph_id": alts[0]["graph_id"]}
    return {"abstain": True,
            "abstain_reason": f"link too weak for {text!r} (no alternative >= {C.LINK_ALT_MIN})"}


def retrieve_node(state: AgentState) -> dict:
    gid = state["disease_graph_id"]
    contract = _call_with_supported(b6.retrieve_candidates, gid, state["split"], state["mode"])
    cands = (contract["candidates"] if isinstance(contract, dict) and "candidates" in contract
             else contract)
    cands = copy.deepcopy(cands)                    # caches READ-ONLY (D-B21) — copy first
    if not cands:
        return {"candidates": [], "abstain": True,
                "abstain_reason": "no 2-hop candidates retrieved (zero-candidate disease)"}
    path_index = {p["path_id"]: p for c in cands for p in c.get("paths", [])}
    return {"candidates": cands, "path_index": path_index}


def _build_candidate_prompt(disease_id, candidate):
    """One small guided-JSON call for a single drug. Its cited_path_ids enum is
    restricted to THIS drug's own paths."""
    paths = candidate.get("paths", [])
    path_ids = [p["path_id"] for p in paths]
    evidence = "\n".join(f"  path {p['path_id']}: {p.get('nl_evidence','')}" for p in paths)
    nm = candidate.get("drug_name") or candidate["drug_graph_id"]
    prompt = (
        f"Disease: {disease_id}\n"
        f"Candidate drug: {candidate['drug_graph_id']} ({nm})\n"
        f"Mechanistic evidence paths:\n{evidence}\n\n"
        "Decide whether these paths support repurposing this drug for the disease. Rules:\n"
        "1. Cite only path ids listed above; never invent one.\n"
        "2. confidence in [0,1]; supported=true only if a path plausibly links drug to disease.\n"
        "Reply as JSON."
    )
    schema = {
        "type": "object",
        "properties": {
            "cited_path_ids": {"type": "array", "maxItems": 3,
                               "items": {"type": "string", "enum": path_ids}},
            "confidence": {"type": "number"},
            "supported": {"type": "boolean"},
        },
        # NB: no free-text field here. xgrammar does not enforce string maxLength,
        # so an unbounded rationale is what truncated generation. Rationales for the
        # C7 plausibility sample are generated separately (best-effort, non-fatal).
        "required": ["cited_path_ids", "confidence", "supported"],
    }
    return prompt, schema


def reason_node(state: AgentState) -> dict:
    reasoner = get_reasoner()
    disease_id = state.get("disease_graph_id")
    per: dict = {}
    items, order = [], []
    for c in state["candidates"][: C.REASON_TOP_K]:
        did = c["drug_graph_id"]
        # STRETCH hook: consult precomputed TxGemma implausible_flag here (OB-8). No-op in core.
        if not c.get("paths"):
            per[did] = {"cited_path_ids": [], "rationale": "", "confidence": 0.0,
                        "supported": False}   # can't reason w/o evidence; empty enum is invalid
            continue
        prompt, schema = _build_candidate_prompt(disease_id, c)
        items.append((prompt, schema, C.REASON_MAX_NEW_TOKENS)); order.append(did)
    # ONE batched call: vLLM runs these concurrently server-side; transformers loops.
    if items:
        for did, res in zip(order, reasoner.generate_json_batch(items)):
            per[did] = res

    if not any(v.get("supported") for v in per.values()):
        return {"reason_out": {"per_candidate": per}, "abstain": True,
                "abstain_reason": "no candidate had faithful mechanistic support"}
    return {"reason_out": {"per_candidate": per}}


def _verify_entry(cited_ids, path_index):
    """MINIMAL faithfulness check (C3 replaces with the full verifier). A citation
    is real iff the path was retrieved AND its id re-derives from its own triples
    via the shared compute_path_id (undirected handled inside it, D-B18/B19)."""
    real, hallucinated = [], []
    for cid in cited_ids:
        p = path_index.get(cid)
        if p and compute_path_id(p["triples"]) == cid:
            real.append(cid)
        else:
            hallucinated.append(cid)
    return real, hallucinated


# Canonical verifier lives in C3; use it when present so the agent and the
# ungrounded baseline are scored by IDENTICAL code (the gap is the headline).
try:
    from c3_verify import verify_citations as _verify_entry  # noqa: F811,E402
except ImportError:
    pass


def verify_node(state: AgentState) -> dict:
    per = state["reason_out"]["per_candidate"]
    path_index = state.get("path_index", {})
    verified, n_hall = [], 0
    for c in state["candidates"][: C.REASON_TOP_K]:
        did = c["drug_graph_id"]
        e = per.get(did, {})
        real, hall = _verify_entry(e.get("cited_path_ids", []), path_index)
        n_hall += len(hall)
        verified.append({
            "drug_graph_id": did,
            "rationale": e.get("rationale", ""),
            "confidence": e.get("confidence"),
            "supported": bool(e.get("supported", False)),
            "cited_path_ids": real,
            "hallucinated_path_ids": hall,
            "faithful": len(real) > 0 and bool(e.get("supported", False)),
        })
    return {"reason_out": {**state["reason_out"], "_verified": verified, "_n_hallucinated": n_hall}}


def assemble_node(state: AgentState) -> dict:
    if state.get("abstain"):
        return {"result": {
            "disease_graph_id": state.get("disease_graph_id"),
            "split": state["split"], "mode": state["mode"],
            "abstain": True, "abstain_reason": state.get("abstain_reason"),
            "ranked_drug_ids": [],
        }}

    verified = state["reason_out"].get("_verified", [])
    faithful = sorted((v for v in verified if v["faithful"]),
                      key=lambda v: (v["confidence"] or 0.0), reverse=True)
    unfaithful = [v for v in verified if not v["faithful"]]
    ordered = [v["drug_graph_id"] for v in faithful] + [v["drug_graph_id"] for v in unfaithful]

    seen = set(ordered)                              # keep rest of pool (B4 order) for recall
    for c in state["candidates"]:
        d = c["drug_graph_id"]
        if d not in seen:
            ordered.append(d)
            seen.add(d)

    return {"result": {
        "disease_graph_id": state.get("disease_graph_id"),
        "split": state["split"], "mode": state["mode"],
        "abstain": False,
        "ranking": verified,
        "ranked_drug_ids": ordered,
        "n_reasoned": len(verified),
        "n_faithful": len(faithful),
        "n_hallucinated_citations": state["reason_out"].get("_n_hallucinated", 0),
        "n_pool": len(state["candidates"]),
    }}


# =====================================================================
# Graph wiring
# =====================================================================
def _route_or_abstain(state: AgentState, nxt: str) -> str:
    return "assemble" if state.get("abstain") else nxt


def build_graph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(AgentState)
    g.add_node("link", link_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("reason", reason_node)
    g.add_node("verify", verify_node)
    g.add_node("assemble", assemble_node)
    g.set_entry_point("link")
    g.add_conditional_edges("link", lambda s: _route_or_abstain(s, "retrieve"),
                            {"retrieve": "retrieve", "assemble": "assemble"})
    g.add_conditional_edges("retrieve", lambda s: _route_or_abstain(s, "reason"),
                            {"reason": "reason", "assemble": "assemble"})
    g.add_conditional_edges("reason", lambda s: _route_or_abstain(s, "verify"),
                            {"verify": "verify", "assemble": "assemble"})
    g.add_edge("verify", "assemble")
    g.add_edge("assemble", END)
    return g.compile()


_APP = None

def _app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def run_agent(disease_graph_id: str | None = None, disease_text: str | None = None,
              split: str = C.HEADLINE_SPLIT, mode: str = "plus_true") -> dict:
    init: AgentState = {"disease_graph_id": disease_graph_id, "disease_text": disease_text,
                        "split": split, "mode": mode, "abstain": False, "abstain_reason": None}
    return _app().invoke(init)["result"]


# =====================================================================
# Harness adapter — frozen interface rank(disease, universe) -> list[str].
# C5 finalizes how split/mode reach the ranker (the harness owns the universe).
# split/mode are constructor-set here; do NOT wire into evaluate() yet.
# =====================================================================
class GroundedAgentRanker:
    mode_aware = True

    def __init__(self, split: str = C.HEADLINE_SPLIT, mode: str = "plus_true"):
        self.split, self.mode = split, mode

    def rank(self, disease: str, universe: set[str]) -> list[str]:
        out = run_agent(disease_graph_id=disease, split=self.split, mode=self.mode)
        if out.get("abstain"):
            return []                                # honest miss; abstention tracked separately (C8)
        ranked = [d for d in out["ranked_drug_ids"] if d in universe]
        seen = set(ranked)
        ranked += [d for d in universe if d not in seen]
        return ranked


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease-id", default="MONDO_0005148")   # type 2 diabetes
    ap.add_argument("--disease-text", default=None)
    ap.add_argument("--split", default=C.HEADLINE_SPLIT)
    ap.add_argument("--mode", default="plus_true")
    a = ap.parse_args()
    res = run_agent(disease_graph_id=a.disease_id, disease_text=a.disease_text,
                    split=a.split, mode=a.mode)
    print(json.dumps(res, indent=2)[:4000])
