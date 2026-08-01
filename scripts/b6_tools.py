#!/usr/bin/env python3
"""
ReTraceRx Phase B — B6: the three agent tools, packaged.

Phase C imports these:
    from b6_tools import link_entity, retrieve_candidates, score_pair

They are thin, DETERMINISTIC, CACHED, LOGGED wrappers over B2/B4/B5:
  * link_entity(text)                       -> query_disease block            (B2)
  * retrieve_candidates(disease_id, k, split)-> full v1.0 contract dict        (B5)
  * score_pair(drug_id, disease_id, split)  -> raw KGE plausibility | None     (B4)

Purity + logging matter (how-to B6): same input -> same output (SapBERT/KGE run in
eval mode, no dropout), and every call is appended to logs/b6_calls.jsonl so the
agent's reliability and the eval harness are auditable.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import b2_link
import b4_rank
import b5_serialize

DEFAULT_SPLIT = "disease_cold"
LOG = Path("logs/b6_calls.jsonl")
LOG.parent.mkdir(parents=True, exist_ok=True)
_CACHE: dict = {}


def _log(tool: str, args: dict, summary: dict):
    rec = {"ts": round(time.time(), 3), "tool": tool, "args": args, "summary": summary}
    with LOG.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _memo(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def link_entity(text: str) -> dict:
    out = _memo(("link", text), lambda: b2_link.link_entity(text))
    _log("link_entity", {"text": text},
         {"graph_id": out["graph_id"], "link_score": out["link_score"],
          "n_alternatives": len(out["alternatives"])})
    return out


def score_pair(drug_id: str, disease_id: str, split: str = DEFAULT_SPLIT) -> float | None:
    out = _memo(("score", drug_id, disease_id, split),
                lambda: b4_rank.score_pair(drug_id, disease_id, split))
    _log("score_pair", {"drug": drug_id, "disease": disease_id, "split": split},
         {"kge_score": out})
    return out


def retrieve_candidates(disease_id: str, k: int = 20, split: str = DEFAULT_SPLIT) -> dict:
    """disease_id = the VERBATIM graph_id (from link_entity). Returns the v1.0 contract."""
    out = _memo(("retrieve", disease_id, k, split),
                lambda: b5_serialize.build_contract(disease_id, is_id=True, split=split, k=k))
    _log("retrieve_candidates", {"disease": disease_id, "k": k, "split": split},
         {"n_candidates": len(out["candidates"])})
    return out


def link_and_retrieve(text: str, k: int = 20, split: str = DEFAULT_SPLIT) -> dict:
    """Convenience: link free text then retrieve in one call (full contract)."""
    return _memo(("link_retrieve", text, k, split),
                 lambda: b5_serialize.build_contract(text, is_id=False, split=split, k=k))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="type 2 diabetes")
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    a = ap.parse_args()
    q = link_entity(a.text)
    print("link_entity:", {k: q[k] for k in ("graph_id", "curie", "link_score")})
    if q["graph_id"]:
        c = retrieve_candidates(q["graph_id"], k=5, split=a.split)
        print(f"retrieve_candidates: {len(c['candidates'])} candidates")
        top = c["candidates"][0]
        print("  top:", top["drug_name"], "score", top["score"])
        print("  score_pair(top):",
              score_pair(top["drug_graph_id"], q["graph_id"], a.split))
    print(f"calls logged to {LOG}")
