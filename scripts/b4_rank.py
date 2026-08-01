#!/usr/bin/env python3
"""
ReTraceRx Phase B — B4: rank candidates & prune paths (context engineering).

Consumes B3's retrieve_paths() and produces ranked, pruned candidates ready for B5.
Ranking combines three signals (how-to B4):
    score = W_KGE*kge_norm + W_SUP*support + W_SPEC*specificity
  * kge_norm  — Phase D RotatE plausibility of (drug --INDICATION--> disease),
                min-max normalized WITHIN this disease's candidate pool.
                RAW score is unbounded (RotatE = negative distance), so it CANNOT
                enter the linear combo un-normalized. NULLABLE (transductive miss).
  * support   — distinct independent paths (distinct intermediate-node tuples),
                capped at SUP_CAP. "10 paths through 3 genes" counts as 3, not 10.
  * specificity — best-path quality: rarer predicates (inverse census frequency) +
                shorter paths score higher; Tier-2 ADME edges penalized (logged decision).

KGE facts (Phase D — status doc wins over the how-to):
  * model = artifacts/kge_rotate_<split>/trained_model.pkl (RotatE 256); NEVER retrain (D-D10)
  * relation = d1_meta.json['ind_relation'] == 'DRG-DIS__INDICATION' (HYPHEN; D-D2).
    The Neo4j composite (underscore) is a DIFFERENT artifact — do not cross-use.
  * split-scoped (D-D8): eval on split S must score with model S or it leaks.
  * transductive (D-D6): entity absent from the factory vocab -> score None (kge nullable).

If the KGE artifacts are absent, B4 degrades to graph-only ranking (kge=None for all)
so it still runs — the how-to's "KGE + path count is enough" fallback.
"""
from __future__ import annotations
import json
import math
from functools import lru_cache
from pathlib import Path

import b3_retrieve  # retrieve_paths(disease_id)

# --- ranking weights & pruning (override via data/retrieval/b4_config.json) ---
W_KGE, W_SUP, W_SPEC = 0.6, 0.3, 0.1
SUP_CAP = 5
TIER2_TYPES = {"DRG_GEN__TRANSPORTER", "DRG_GEN__CARRIER",
               "DRG_GEN__SUBSTRATE", "DRG_GEN__ENZYME"}   # ADME (logged decision)
TIER2_PENALTY = 0.5
TOP_K, TOP_M = 20, 3
DEFAULT_SPLIT = "disease_cold"   # headline; MUST match the holdout split under eval

REL_TYPES = Path("data/retrieval/b3_rel_types.json")
KGE_DIR = Path("artifacts")
CONFIG = Path("data/retrieval/b4_config.json")


def _cfg():
    c = dict(W_KGE=W_KGE, W_SUP=W_SUP, W_SPEC=W_SPEC, SUP_CAP=SUP_CAP,
             TIER2_PENALTY=TIER2_PENALTY, TOP_K=TOP_K, TOP_M=TOP_M,
             DEFAULT_SPLIT=DEFAULT_SPLIT)
    if CONFIG.exists():
        c.update(json.loads(CONFIG.read_text()))
    return c


# ---------------------------------------------------------------- specificity
@lru_cache(maxsize=1)
def _pred_specificity() -> dict[str, float]:
    """Inverse-frequency specificity per composite type, scaled to [0,1] over the
    metapath's types. Rarer predicate -> higher specificity."""
    rt = json.loads(REL_TYPES.read_text())
    counts = rt.get("census_counts") or {}
    if not counts:
        print("WARN: no census_counts in b3_rel_types.json — specificity is neutral (0.5).")
        return {}
    logs = {t: math.log(c) for t, c in counts.items() if c and c > 0}
    lo, hi = min(logs.values()), max(logs.values())
    if hi == lo:
        return {t: 1.0 for t in logs}
    return {t: 1.0 - (v - lo) / (hi - lo) for t, v in logs.items()}  # rare->1, common->0


def _path_specificity(path: dict, spec: dict[str, float]) -> float:
    edge_specs = []
    for t in path["rel_types"]:
        s = spec.get(t, 0.5)
        if t in TIER2_TYPES:
            s *= TIER2_PENALTY
        edge_specs.append(s)
    # A path is as specific as its RAREST (most discriminating) edge. Using max, not
    # mean, so the mandatory DIS_GEN__ASSOCIATED_WITH terminus (the single most common
    # type, ~0 specificity, present on EVERY path) can't dilute the drug-side signal.
    peak_edge = max(edge_specs) if edge_specs else 0.0
    length_factor = 2.0 / path["n_hops"]  # 2-hop -> 1.0, 3-hop -> 0.67
    return peak_edge * length_factor


def _path_tier(path: dict) -> int:
    return 2 if any(t in TIER2_TYPES for t in path["rel_types"]) else 1


def _intermediate_key(path: dict) -> tuple:
    # nodes(path) = [disease, ...intermediates..., drug]; independence = intermediate tuple
    return tuple(path["node_ids"][1:-1])


# ---------------------------------------------------------------- KGE scoring
_KGE: dict = {}


def _load_kge(split: str):
    if split in _KGE:
        return _KGE[split]
    model_p = KGE_DIR / f"kge_rotate_{split}" / "trained_model.pkl"
    fac_p = KGE_DIR / "kge_data" / split / "train_factory"
    meta_p = KGE_DIR / "kge_data" / split / "d1_meta.json"
    if not (model_p.exists() and fac_p.exists()):
        _KGE[split] = None  # graph-only degrade
        print(f"WARN: KGE artifacts for split '{split}' not found "
              f"({model_p} / {fac_p}) — ranking graph-only (kge=None).")
        return None
    import torch
    from pykeen.triples import TriplesFactory
    model = torch.load(model_p, map_location="cpu", weights_only=False)
    tf = TriplesFactory.from_path_binary(fac_p)
    ind_rel = "DRG-DIS__INDICATION"
    if meta_p.exists():
        ind_rel = json.loads(meta_p.read_text()).get("ind_relation", ind_rel)
    if ind_rel not in tf.relation_to_id:
        raise RuntimeError(f"indication relation {ind_rel!r} absent from factory vocab "
                           f"(split {split}); check d1_meta.json.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    _KGE[split] = dict(model=model, e2id=tf.entity_to_id,
                       rid=tf.relation_to_id[ind_rel], dev=dev, torch=torch)
    print(f"KGE split={split}: {len(tf.entity_to_id)} entities, relation={ind_rel}")
    return _KGE[split]


def score_pairs(disease_id: str, drug_ids: list[str], split: str) -> dict[str, float | None]:
    """Raw RotatE plausibility of (drug --INDICATION--> disease). None if out-of-vocab."""
    K = _load_kge(split)
    out = {d: None for d in drug_ids}
    if K is None:
        return out
    torch = K["torch"]; e2id = K["e2id"]
    if disease_id not in e2id:
        return out  # disease unseen -> nothing scoreable
    dis = e2id[disease_id]
    triples, keys = [], []
    for d in drug_ids:
        if d in e2id:
            triples.append([e2id[d], K["rid"], dis]); keys.append(d)
    if not triples:
        return out
    with torch.no_grad():
        t = torch.as_tensor(triples, dtype=torch.long, device=K["dev"])
        s = K["model"].score_hrt(t).squeeze(-1).cpu().tolist()
    for d, sc in zip(keys, s):
        out[d] = float(sc)
    return out


def score_pair(drug_id: str, disease_id: str, split: str = DEFAULT_SPLIT) -> float | None:
    """Phase B tool (B6 signature). Single-pair convenience over score_pairs."""
    return score_pairs(disease_id, [drug_id], split)[drug_id]


# ---------------------------------------------------------------- ranking
def _minmax(vals: dict[str, float | None]) -> dict[str, float | None]:
    present = [v for v in vals.values() if v is not None]
    if not present:
        return {k: None for k in vals}
    lo, hi = min(present), max(present)
    if hi == lo:
        return {k: (1.0 if v is not None else None) for k, v in vals.items()}
    return {k: ((v - lo) / (hi - lo) if v is not None else None) for k, v in vals.items()}


def rank_candidates(disease_id: str, split: str | None = None,
                    k: int | None = None, m: int | None = None) -> list[dict]:
    cfg = _cfg()
    split = split or cfg["DEFAULT_SPLIT"]
    k = k or cfg["TOP_K"]; m = m or cfg["TOP_M"]
    spec = _pred_specificity()

    cands = b3_retrieve.retrieve_paths(disease_id)
    if not cands:
        return []

    # KGE (batched over all candidate drugs) then min-max within this pool
    drug_ids = [c["drug_graph_id"] for c in cands]
    raw_kge = score_pairs(disease_id, drug_ids, split)
    norm_kge = _minmax(raw_kge)

    for c in cands:
        gid = c["drug_graph_id"]
        # annotate each path with specificity + tier
        for p in c["paths"]:
            p["path_specificity"] = _path_specificity(p, spec)
            p["tier"] = _path_tier(p)
        # support = distinct independent paths (distinct intermediate tuples)
        indep = {_intermediate_key(p) for p in c["paths"]}
        c["support_count"] = len(indep)
        support = min(len(indep), cfg["SUP_CAP"]) / cfg["SUP_CAP"]
        # specificity = best path
        c["specificity"] = max(p["path_specificity"] for p in c["paths"])
        c["kge_score"] = raw_kge[gid]                    # RAW (contract; nullable)
        kn = norm_kge[gid]
        if kn is not None:
            c["score"] = round(cfg["W_KGE"] * kn + cfg["W_SUP"] * support
                               + cfg["W_SPEC"] * c["specificity"], 6)
        else:
            # renormalize the remaining weights to sum to 1 (kge absent)
            wsum = cfg["W_SUP"] + cfg["W_SPEC"]
            c["score"] = round((cfg["W_SUP"] * support
                                + cfg["W_SPEC"] * c["specificity"]) / wsum, 6)
        c["_kge_norm"] = kn
        c["_support_frac"] = support
        # prune to top-M paths by specificity
        c["paths"] = sorted(c["paths"], key=lambda p: p["path_specificity"],
                            reverse=True)[:m]

    ranked = sorted(cands, key=lambda c: c["score"], reverse=True)[:k]
    for i, c in enumerate(ranked, 1):
        c["rank"] = i
    return ranked


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("disease_id")
    ap.add_argument("--split", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--m", type=int, default=None)
    a = ap.parse_args()
    out = rank_candidates(a.disease_id, split=a.split, k=a.k, m=a.m)
    n_kge = sum(c["kge_score"] is not None for c in out)
    print(f"{a.disease_id}: {len(out)} ranked candidates "
          f"({n_kge} KGE-scored, {len(out)-n_kge} null)")
    for c in out:
        kge = f"{c['kge_score']:.3f}" if c["kge_score"] is not None else "  null"
        print(f"  #{c['rank']:>2} {c['drug_graph_id']:16s} {str(c['drug_name'])[:24]:24s} "
              f"score={c['score']:.4f} | kge={kge} sup={c['support_count']} "
              f"spec={c['specificity']:.3f} | paths={len(c['paths'])} tier{min(p['tier'] for p in c['paths'])}")
