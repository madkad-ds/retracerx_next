#!/usr/bin/env python3
"""
e2_api.py — FastAPI wrapper around the Phase C agent (E2).

Endpoints over the warm container stack (Neo4j + vLLM):
  GET  /health     — readiness + which backend/model the app is bound to
  POST /link       — free text → contract query_disease block (graph_id + alternatives)
  POST /repurpose  — link-once → dispatch grounded agent OR ungrounded baseline

Launch (in a shell where `source scripts/serve_env.sh` has run, so
REASONER_BACKEND=vllm + VLLM_* are exported and PYTHONPATH includes scripts):

    uvicorn e2_api:app --host 127.0.0.1 --port 8080

Bound to the ACTUAL Phase C signatures (the status record wins over the how-to):
  * run_agent(disease_graph_id=None, disease_text=None, split=…, mode=…)  [c2_agent]
  * run_ungrounded(disease_graph_id, n=10)                                [c6_ungrounded]  ← demo contrast
  * run_ungrounded_pool(disease_graph_id, split, mode)                    [c6_ungrounded]  ← fair within-pool
  * link_entity(text) -> {graph_id, curie, link_score, alternatives}      [b6_tools]

E2 constraints honoured: load once (import + warm); deep-copy results (B6 caches
are READ-ONLY, D-B21); per-request timeout so a slow Neo4j path can't hang a
call; determinism inherited from the agent (greedy + seed 20260727, D-C14).
"""
from __future__ import annotations
import asyncio, copy, csv, functools, os, sys, time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_config as C

# ─── entrypoints (verified against the real signatures) ──────────────────────
import b6_tools as b6
from c2_agent import run_agent
from c6_ungrounded import run_ungrounded, run_ungrounded_pool
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_TIMEOUT_S = float(os.environ.get("RETRACERX_REQUEST_TIMEOUT_S", "180"))
WARM_TIMEOUT_S    = float(os.environ.get("RETRACERX_WARM_TIMEOUT_S", "120"))
WARM_ON           = os.environ.get("RETRACERX_WARM", "1") == "1"
WARM_GRAPH_ID     = os.environ.get("RETRACERX_WARM_GRAPH_ID", "MONDO_0005148")  # T2D probe


async def _call(fn, *args, **kwargs):
    """Offload a blocking agent/tool call to a worker thread with a timeout, and
    return a defensive deep copy (B6 caches are read-only, D-B21)."""
    loop = asyncio.get_event_loop()
    out = await asyncio.wait_for(
        loop.run_in_executor(None, functools.partial(fn, *args, **kwargs)),
        timeout=REQUEST_TIMEOUT_S)
    return copy.deepcopy(out)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.backend = os.environ.get("REASONER_BACKEND", "")
    app.state.model = os.environ.get("VLLM_SERVED_MODEL_NAME") or C.resolve_served_model_name()
    app.state.warnings = []
    app.state.ready = False

    if app.state.backend != "vllm":
        app.state.warnings.append(
            f"REASONER_BACKEND={app.state.backend!r} (expected 'vllm'; in-proc "
            f"transformers is minutes-per-disease, E-C9). Did you source serve_env.sh?")
    if not os.environ.get("VLLM_API_KEY"):
        app.state.warnings.append("VLLM_API_KEY not set — the agent may 401 against vLLM.")

    # Warm ONCE so the agent singletons (Neo4j driver, vLLM client, graph,
    # verifier) init before serving. Uses the proven c2_smoke probe.
    if WARM_ON:
        try:
            await _call(run_agent, disease_graph_id=WARM_GRAPH_ID,
                        split=C.HEADLINE_SPLIT, mode="plus_true")
        except Exception as e:  # noqa: BLE001 — never die on a warm miss
            app.state.warnings.append(f"warm skipped: {type(e).__name__}: {e}")

    app.state.ready = True
    yield


app = FastAPI(title="ReTraceRx API", version="E2", lifespan=lifespan)


class LinkReq(BaseModel):
    disease_text: str


class Req(BaseModel):
    disease_text: str | None = None
    disease_graph_id: str | None = None       # if the UI already disambiguated
    split: str = C.HEADLINE_SPLIT
    mode: str = "whitelist"                    # deployment framing (live demo has no true drug)
    grounded: bool = True
    ungrounded_variant: str = "free"           # "free" (demo contrast) | "pool" (fair within-pool)
    n: int = 10                                # free-ungrounded proposal cap
    top_k: int | None = None                   # pool-ungrounded pool cap


@app.get("/health")
def health():
    return {
        "ok": bool(getattr(app.state, "ready", False)),
        "backend": getattr(app.state, "backend", None),
        "model": getattr(app.state, "model", None),
        "warnings": getattr(app.state, "warnings", []),
    }


@app.post("/link")
async def link(r: LinkReq):
    try:
        return await _call(b6.link_entity, r.disease_text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@functools.lru_cache(maxsize=1)
def _name_map() -> dict:
    """drug_graph_id → human name, from the reachable whitelist (D-C10 keys).
    Fallback only; retrieved candidates usually carry drug_name directly."""
    m = {}
    try:
        with open(C.WHITELIST_CSV, newline="") as f:
            for row in csv.DictReader(f):
                gid = row.get("okg_drug_id")
                if gid:
                    m[gid] = row.get("name") or gid
    except Exception:  # noqa: BLE001
        pass
    return m


@app.post("/contract")
async def contract(r: Req):
    """Inspection endpoint: the raw retrieval contract for a disease (candidates
    with their paths — path_id, triples, nl_evidence, sources). The UI uses this
    to resolve the agent's cited_path_ids into renderable evidence."""
    gid = r.disease_graph_id
    if gid is None:
        if not r.disease_text:
            raise HTTPException(400, "provide disease_text or disease_graph_id")
        gid = (await _call(b6.link_entity, r.disease_text)).get("graph_id")
        if not gid:
            raise HTTPException(422, "disease did not link confidently")
    try:
        return await _call(b6.retrieve_candidates, gid, split=r.split)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/repurpose")
async def repurpose(r: Req):
    if r.mode not in ("whitelist", "plus_true"):
        raise HTTPException(400, f"mode must be 'whitelist' or 'plus_true', got {r.mode!r}")
    if r.split not in C.SPLITS:
        raise HTTPException(400, f"split must be one of {C.SPLITS}, got {r.split!r}")
    if not (r.disease_text or r.disease_graph_id):
        raise HTTPException(400, "provide disease_text or disease_graph_id")

    t0 = time.time()
    query_disease = None
    gid = r.disease_graph_id
    try:
        # 1. resolve the disease id ONCE (unless the UI already did)
        if gid is None:
            query_disease = await _call(b6.link_entity, r.disease_text)
            gid = query_disease.get("graph_id")
            if not gid:
                # low-confidence link (D-B7): don't guess — let the UI disambiguate
                return {"needs_disambiguation": True, "query_disease": query_disease,
                        "_meta": {"latency_s": round(time.time() - t0, 2)}}

        # 2. dispatch
        if r.grounded:
            out = await _call(run_agent, disease_graph_id=gid, split=r.split, mode=r.mode)
        elif r.ungrounded_variant == "pool":
            out = await _call(run_ungrounded_pool, gid, split=r.split, mode=r.mode, top_k=r.top_k)
        else:
            out = await _call(run_ungrounded, gid, n=r.n)
    except asyncio.TimeoutError:
        raise HTTPException(504, f"agent exceeded {REQUEST_TIMEOUT_S:.0f}s (slow path lookup?)")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")

    if not isinstance(out, dict):
        out = {"result": out}
    # attach human names for the ranked drugs (UI has only CHEMBL ids otherwise)
    if r.grounded and isinstance(out.get("ranking"), list):
        nm = _name_map()
        for cand in out["ranking"]:
            gid_c = cand.get("drug_graph_id")
            if gid_c and "drug_name" not in cand:
                cand["drug_name"] = nm.get(gid_c, gid_c)
    meta = out.setdefault("_meta", {})
    meta.update(grounded=r.grounded, mode=r.mode, split=r.split,
                disease_graph_id=gid, latency_s=round(time.time() - t0, 2))
    if not r.grounded:
        meta["ungrounded_variant"] = r.ungrounded_variant
    if query_disease is not None:
        meta["query_disease"] = query_disease   # link_score + alternatives for the UI
    return out
