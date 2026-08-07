#!/usr/bin/env python3
"""
c3_verify.py — Phase C faithfulness verifier (the headline deliverable).

ONE verifier, imported by BOTH the grounded agent (C2) and the ungrounded baseline
(C6), so the grounded-vs-ungrounded hallucination gap is measured by identical code.

Core check (how-to C3): a citation is REAL iff the cited path was retrieved AND its
path_id re-derives from its own triples via the SHARED compute_path_id (never
reimplemented; D-B18). Matching is UNDIRECTED: the 13 undirected labels are stored
single-orientation (A3 D16), and compute_path_id normalizes this — so re-deriving
from the stored triples is the undirected check. A directed re-check would falsely
reject half the labels; if you additionally probe Neo4j, use `-[r]-` (see
`edge_exists_undirected`).

Hallucination has two granularities (both reported):
  - CLAIM-level  = cited path_ids that don't verify / total cited path_ids
  - PICK-level   = drug picks with NO verified path / total drug picks
                   (a pick with no grounding is "a claim not tied to a path_id",
                    how-to C3). PICK-level is the cross-system headline: the
                    ungrounded model has zero real citations, so its pick-level
                    rate is ~1.0 by construction; the grounded agent's is ~0.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random  # noqa: E402
from b5_serialize import compute_path_id  # noqa: E402 — shared identity fn (D-B18)


def _index(retrieved_paths) -> dict:
    """Accept either a list of path objects or a prebuilt {path_id: path} dict."""
    if isinstance(retrieved_paths, dict):
        return retrieved_paths
    return {p["path_id"]: p for p in retrieved_paths}


def verify_citations(cited_ids, retrieved_paths):
    """Return (real, hallucinated). `retrieved_paths` = a disease's
    candidates[].paths[] (list) or a {path_id: path} index."""
    valid = _index(retrieved_paths)
    real, hallucinated = [], []
    for cid in cited_ids:
        p = valid.get(cid)
        # integrity: id must recompute from its own triples (undirected, D-B19)
        if p and compute_path_id(p["triples"]) == cid:
            real.append(cid)
        else:
            hallucinated.append(cid)
    return real, hallucinated


@dataclass
class FaithfulnessReport:
    n_picks: int = 0
    n_faithful: int = 0                 # picks with >= 1 verified path
    n_real_citations: int = 0
    n_hallucinated_citations: int = 0
    per_pick: list = field(default_factory=list)

    @property
    def pick_level_hallucination(self) -> float:
        return 0.0 if self.n_picks == 0 else (self.n_picks - self.n_faithful) / self.n_picks

    @property
    def claim_level_hallucination(self) -> float:
        tot = self.n_real_citations + self.n_hallucinated_citations
        return 0.0 if tot == 0 else self.n_hallucinated_citations / tot

    def as_dict(self) -> dict:
        return {"n_picks": self.n_picks, "n_faithful": self.n_faithful,
                "n_real_citations": self.n_real_citations,
                "n_hallucinated_citations": self.n_hallucinated_citations,
                "pick_level_hallucination": round(self.pick_level_hallucination, 4),
                "claim_level_hallucination": round(self.claim_level_hallucination, 4)}


def score_system(picks, retrieved_paths) -> FaithfulnessReport:
    """`picks` = [{drug_graph_id, cited_path_ids (RAW, pre-verify)}...] for one
    disease. Verifies each pick's citations against the disease's retrieved paths
    and rolls up both hallucination granularities. Works identically for the
    grounded agent (has real citations) and the ungrounded baseline (cited=[])."""
    rep = FaithfulnessReport()
    idx = _index(retrieved_paths)
    for pk in picks:
        real, hall = verify_citations(pk.get("cited_path_ids", []), idx)
        faithful = len(real) > 0
        rep.n_picks += 1
        rep.n_faithful += int(faithful)
        rep.n_real_citations += len(real)
        rep.n_hallucinated_citations += len(hall)
        rep.per_pick.append({"drug_graph_id": pk.get("drug_graph_id"),
                             "real": real, "hallucinated": hall, "faithful": faithful})
    return rep


# --- optional: self-correction (how-to C3) ---------------------------------
def self_correct(reasoner, prompt, schema, retrieved_paths, rounds: int = 1,
                 max_new_tokens: int = 128):
    """Generate, verify, and (if any citation is hallucinated) re-prompt to drop
    the bad ids and re-emit. Returns (final_obj, pre_report, post_report). For the
    GUIDED grounded agent this is usually a no-op (citations are enum-constrained);
    it matters when guided decoding is OFF (the unconstrained rate, how-to C4)."""
    obj = reasoner.generate_json(prompt, schema, max_new_tokens=max_new_tokens)
    pre = score_system([obj], retrieved_paths)
    if pre.n_hallucinated_citations == 0 or rounds <= 0:
        return obj, pre, pre
    bad = pre.per_pick[0]["hallucinated"]
    fix_prompt = (prompt + f"\n\nThese cited path ids were NOT retrieved and must be "
                  f"removed: {bad}. Re-answer citing only valid retrieved path ids.")
    obj2 = reasoner.generate_json(fix_prompt, schema, max_new_tokens=max_new_tokens)
    post = score_system([obj2], retrieved_paths)
    return obj2, pre, post


# --- optional: Neo4j undirected edge re-check (defense-in-depth) ------------
def edge_exists_undirected(session, head: str, tail: str, neo4j_type: str) -> bool:
    """Confirm an edge exists in the graph regardless of stored orientation.
    Uses `-[r]-` (undirected) because 13 labels are single-orientation (D16).
    `neo4j_type` must be the UNDERSCORE form (c0_config.to_neo4j_type)."""
    q = (f"MATCH (h {{id:$h}})-[r:`{neo4j_type}`]-(t {{id:$t}}) "
         f"RETURN count(r) > 0 AS ok")
    rec = session.run(q, h=head, t=tail).single()
    return bool(rec and rec["ok"])


if __name__ == "__main__":
    # tiny self-test with a fabricated vs real citation
    paths = [{"path_id": None, "triples": [["A", "R__X", "B"], ["B", "S__Y", "C"]]}]
    paths[0]["path_id"] = compute_path_id(paths[0]["triples"])
    real, hall = verify_citations([paths[0]["path_id"], "p_deadbeef00"], paths)
    print("real:", real, "hallucinated:", hall)
    rep = score_system([{"drug_graph_id": "CHEMBL_1",
                         "cited_path_ids": [paths[0]["path_id"], "p_deadbeef00"]}], paths)
    print(rep.as_dict())


def sampled_graph_assert(all_paths, sample_rate: float = 0.25, seed: int = 20260727) -> int:
    """SAMPLED third verifier check (run every eval, high rate). Re-check that a
    fraction of retrieved path edges actually exist in Neo4j under UNDIRECTED
    matching (-[r]-). Catches a denormalized/cached path that verifies by hash but
    isn't a real edge (the 'read the artifact, not the sentence' failure mode).
    Raises AssertionError on any mismatch; returns the number of paths checked."""
    import os
    import c0_config as C
    paths = [p for p in all_paths if p.get("triples")]
    if not paths:
        return 0
    random.Random(seed).shuffle(paths)
    sample = paths[: max(1, int(len(paths) * sample_rate))]
    try:
        from neo4j import GraphDatabase
    except Exception as e:  # noqa: BLE001
        print(f"[sampled_graph_assert] neo4j driver unavailable, skipped: {e}")
        return 0
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "retracerx"))
    drv = GraphDatabase.driver(uri, auth=auth)
    bad = []
    with drv.session() as sess:
        for p in sample:
            for h, rel, t in p["triples"]:
                if not edge_exists_undirected(sess, h, t, C.to_neo4j_type(rel)):
                    bad.append((p.get("path_id"), h, rel, t))
    drv.close()
    if bad:
        raise AssertionError(f"{len(bad)} sampled path edge(s) not in graph (undirected): "
                             f"{bad[:5]}")
    return len(sample)
