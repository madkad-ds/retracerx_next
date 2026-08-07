#!/usr/bin/env python3
"""c5_rankers.py — real bodies for the harness's deferred rankers.

Constructor is (split, device) to match d4_harness.main()'s
`RANKERS[args.ranker](split, device)` (same shape as KGERanker/MockRanker).
device is accepted but unused — the agent talks to the in-process reasoner.

rank(disease, universe) -> best-first ids covering the universe (D-D16):
  - rank the universe you're HANDED (don't re-retrieve it),
  - DETERMINISTIC tail = sorted(universe - ranked) (str-hash randomization would
    otherwise make ranks non-reproducible across the harness's per-disease cache),
  - cache per (disease, split): reasoning is universe-independent, so one agent run
    serves both candidate_modes.
Abstention returns [] -> the harness scores rank=None (a miss), which is correct.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c0_config as C  # noqa: E402
from c2_agent import run_agent  # noqa: E402
from c6_ungrounded import run_ungrounded  # noqa: E402
import b4_rank  # noqa: E402

_AGENT: dict = {}
_UFREE: dict = {}
_GRAPHRAG: dict = {}


def _cover(order, universe):
    u = set(universe)
    ranked, seen = [], set()
    for d in order:
        if d in u and d not in seen:
            ranked.append(d)
            seen.add(d)
    ranked.extend(sorted(u - seen))          # deterministic tail (never raw set order)
    return ranked


class GroundedAgentRanker:
    name = "grounded_agent"

    def __init__(self, split, device=None):
        self.split = split

    def rank(self, disease, universe):
        key = (disease, self.split)
        if key not in _AGENT:
            _AGENT[key] = run_agent(disease_graph_id=disease, split=self.split)
        out = _AGENT[key]
        return [] if out.get("abstain") else _cover(out.get("ranked_drug_ids", []), universe)


class UngroundedLLMRanker:
    name = "ungrounded_llm"

    def __init__(self, split, device=None, n: int = 10):
        self.split, self.n = split, n

    def rank(self, disease, universe):
        if disease not in _UFREE:
            _UFREE[disease] = run_ungrounded(disease, self.n)
        return _cover(_UFREE[disease].get("ranked_drug_ids", []), universe)


class GraphRAGRanker:
    """B4-only baseline: retrieved pool in B4 composite-score order, NO reasoner.
    Isolates the LLM re-ranking value-add (agent vs graph_rag = does reasoning
    beat B4's order on the same pool). recall@K <= B pool-recall by construction."""
    name = "graph_rag"

    def __init__(self, split, device=None):
        self.split = split

    def rank(self, disease, universe):
        key = (disease, self.split)
        if key not in _GRAPHRAG:
            cands = b4_rank.rank_candidates(disease, split=self.split)
            _GRAPHRAG[key] = [c["drug_graph_id"] for c in cands]
        return _cover(_GRAPHRAG[key], universe)


def pseudo_scores(ranked_ids):
    """Rank-derived pseudo-score (1/rank) — the SINGLE convention for AUROC/D5b on
    ALL non-KGE systems (agent, ungrounded, graph_rag), since none emits a
    calibrated triple score. Apply identically; state the calibration caveat in the
    writeup. Computable straight from the harness per-query `rank` column too."""
    return {d: 1.0 / (i + 1) for i, d in enumerate(ranked_ids)}
