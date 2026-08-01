#!/usr/bin/env python3
"""
ReTraceRx Phase B — B2.2: entity linker (free-text disease -> OptimusKG node).

Three escalating layers — exact -> fuzzy -> SapBERT/FAISS embedding — returning the
FROZEN retrieval-contract v1.0 `query_disease` block:

    {"text","graph_id","curie","link_score","alternatives":[{text,graph_id,curie,link_score}]}

Key facts baked in (from Phase A + our B1/contract decisions):
  * graph_id = VERBATIM OptimusKG id (Neo4j / B3 key);  curie = colon-form DISPLAY only.
  * B1 `src` scope tags fold into link_score: name > exact-synonym > concept/snomed name.
  * EFO/MONDO (and other) ambiguity is RETURNED as `alternatives` (option 1), never merged.
  * Low-confidence embedding hits -> graph_id=None + candidates in alternatives to disambiguate.

Signature note: this SUPERSEDES the how-to's `link_entity -> tuple[str|None,float]`.
It returns the richer contract block, which feeds Phase C directly.

Load is lazy and cached; the SapBERT encoder loads only on first embedding fallback.
Every call is logged (B6 requirement) via the module logger 'retracerx.b2'.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl
import faiss
from rapidfuzz import process, fuzz

META = Path("data/retrieval/faiss_disease_meta.parquet")
IDX = Path("data/retrieval/faiss_disease.index")
MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DIM = 768

# scope base scores (exact-layer ceiling per surface provenance)
SCOPE = {"name": 1.00, "synonym:exact": 0.98, "concept_name": 0.95, "snomed_full_name": 0.95}
FUZZY_CAP = 0.95            # fuzzy link_score can't reach exact territory
EMB_THRESHOLD = 0.85       # below this, an embedding-only hit is "unconfident"
ALT_MARGIN = 0.03          # alternatives = candidates within this of the best
FUZZY_MIN = 90             # min token_sort_ratio to accept the fuzzy layer
K = 5                      # embedding neighbours to pull
MAX_ALTS = 5

log = logging.getLogger("retracerx.b2")
log.addHandler(logging.NullHandler())

_WS = re.compile(r"\s+")
_STATE: dict = {}
_ENC: dict = {}


def norm_surface(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return _WS.sub(" ", s)


def _load() -> dict:
    if _STATE:
        return _STATE
    if not (META.exists() and IDX.exists()):
        raise FileNotFoundError("Run b2_build_index.py first (missing FAISS index/meta).")
    meta = pl.read_parquet(META)
    index = faiss.read_index(str(IDX))
    if index.d != DIM or index.ntotal != meta.height:
        raise RuntimeError("index/meta mismatch — rebuild with b2_build_index.py")
    sn = meta["surface_norm"].to_list()
    exact: dict[str, list[int]] = {}
    for i, s in enumerate(sn):
        exact.setdefault(s, []).append(i)
    names = {gid: surf for gid, surf in
             meta.filter(pl.col("src") == "name").select(["graph_id", "surface"]).iter_rows()}
    _STATE.update(
        meta=meta, index=index, exact=exact, names=names,
        gid=meta["graph_id"].to_list(), cur=meta["curie"].to_list(),
        src=meta["src"].to_list(), surf=meta["surface"].to_list(),
        uniq=list(exact.keys()),  # unique normalized surfaces for fuzzy
    )
    return _STATE


def _encoder():
    if _ENC:
        return _ENC
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _ENC.update(torch=torch, dev=dev,
                tok=AutoTokenizer.from_pretrained(MODEL),
                enc=AutoModel.from_pretrained(MODEL).to(dev).eval())
    return _ENC


def _embed(text: str) -> np.ndarray:
    e = _encoder(); torch = e["torch"]
    t = e["tok"]([text], padding=True, truncation=True, max_length=32,
                 return_tensors="pt").to(e["dev"])
    with torch.no_grad():
        v = e["enc"](**t).last_hidden_state[:, 0, :]
        v = torch.nn.functional.normalize(v, dim=1)
    return v.cpu().numpy().astype("float32")


def _upd(cand: dict, S: dict, idx: int, score: float, via: str):
    gid = S["gid"][idx]
    cur = cand.get(gid)
    if cur is None or score > cur["score"]:
        cand[gid] = {"graph_id": gid, "curie": S["cur"][idx],
                     "name": S["names"].get(gid, S["surf"][idx]),
                     "score": score, "via": via}


def link_entity(text: str, k: int = K, emb_threshold: float = EMB_THRESHOLD,
                margin: float = ALT_MARGIN, fuzzy_min: int = FUZZY_MIN,
                max_alts: int = MAX_ALTS) -> dict:
    S = _load()
    q = norm_surface(text)
    cand: dict[str, dict] = {}
    via = "none"

    # --- Layer 1: exact (normalized) ---
    if q in S["exact"]:
        via = "exact"
        for i in S["exact"][q]:
            _upd(cand, S, i, SCOPE.get(S["src"][i], 0.90), f"exact/{S['src'][i]}")
    else:
        # --- Layer 2: fuzzy over unique normalized surfaces ---
        hits = process.extract(q, S["uniq"], scorer=fuzz.token_sort_ratio, limit=10)
        good = [(s, ratio) for s, ratio, _ in hits if ratio >= fuzzy_min]
        if good:
            via = "fuzzy"
            for s, ratio in good:
                for i in S["exact"][s]:
                    base = SCOPE.get(S["src"][i], 0.90)
                    _upd(cand, S, i, min(FUZZY_CAP, (ratio / 100.0) * base),
                         f"fuzzy/{S['src'][i]}")
        else:
            # --- Layer 3: embedding fallback ---
            via = "embedding"
            D, I = S["index"].search(_embed(text), k)
            for score, idx in zip(D[0], I[0]):
                if idx >= 0:
                    _upd(cand, S, int(idx), float(score), f"embedding/{S['src'][int(idx)]}")

    ranked = sorted(cand.values(), key=lambda c: c["score"], reverse=True)
    if not ranked:
        out = {"text": text, "graph_id": None, "curie": None,
               "link_score": 0.0, "alternatives": []}
        log.info("link_entity via=none text=%r -> UNLINKED", text)
        return out

    best = ranked[0]
    confident = not (via == "embedding" and best["score"] < emb_threshold)

    def as_alt(c):
        return {"text": c["name"], "graph_id": c["graph_id"],
                "curie": c["curie"], "link_score": round(c["score"], 4)}

    if confident:
        alts = [as_alt(c) for c in ranked[1:] if c["score"] >= best["score"] - margin][:max_alts]
        out = {"text": text, "graph_id": best["graph_id"], "curie": best["curie"],
               "link_score": round(best["score"], 4), "alternatives": alts}
    else:
        # unconfident: no primary link; surface all candidates for the agent
        out = {"text": text, "graph_id": None, "curie": None,
               "link_score": round(best["score"], 4),
               "alternatives": [as_alt(c) for c in ranked[:max_alts]]}

    log.info("link_entity via=%s text=%r -> %s score=%.4f alts=%d",
             via, text, out["graph_id"], out["link_score"], len(out["alternatives"]))
    return out


def _selftest():
    for qtext in ["type 2 diabetes", "T2DM", "sugar diabetes", "diabetis", "asthma",
                  "nonsense zzzq disease"]:
        print(qtext, "->", json.dumps(link_entity(qtext), ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="disease mention to link")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    if a.debug:
        logging.basicConfig(level=logging.INFO)
    if a.selftest or not a.query:
        _selftest()
    else:
        print(json.dumps(link_entity(a.query), indent=2, ensure_ascii=False))
