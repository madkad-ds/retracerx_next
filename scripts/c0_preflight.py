#!/usr/bin/env python3
"""
c0_preflight.py — Phase C, stage C0 gate (CORE scope).

Verifies every FROZEN input Phase C consumes is present and the right shape,
BEFORE any agent/verifier code runs. Reads only; never mutates.

Checks (HARD unless marked soft):
  1. Contract v1.0 doc + JSON schema, contract_version const == 1.0   (D-B1)
  2. b5_serialize.compute_path_id imports + deterministic             (D-B18)
  3. b6_tools exposes the four tools                                  (D-B21)
  4. d4_harness.evaluate callable; print detected rank() signature    (D-D16/17) [soft]
  5. reachable whitelist == 1,565                                     (A8/D35)
  6. per-split KGE model + d1_meta; ind_relation is the HYPHEN form   (D-D2/C-D2)
  7. core reasoner deps: transformers importable, a guided-JSON lib
     importable (lmfe|outlines|xgrammar), CUDA visible               [soft]
  8. Neo4j up (filtered, 50 TYPEs)                                    [soft]
  9. writable logs/ and data/phaseC/

NOTE: vLLM serving, AWQ quantization, and the TxGemma property tool are Stretch
(phaseC/stretch/); they are intentionally NOT preflight-checked in core.

Run from repo root:
    export PATH="$HOME/.local/bin:$PATH"
    source .venv-data/bin/activate && source scripts/okg_env.sh
    source scripts/neo4j_env.sh
    export PYTHONPATH="$PWD/scripts"
    python phaseC/scripts/c0_preflight.py
"""
from __future__ import annotations
import importlib
import importlib.util
import inspect
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c0_config as C  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, ok: bool, hard: bool, detail: str = "") -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "hard": hard, "detail": detail})
    print(f"[{('PASS' if ok else ('FAIL' if hard else 'WARN')):4}] {name}: {detail}")


def check_contract() -> None:
    md_ok = C.CONTRACT_MD.is_file()
    sc_ok = C.CONTRACT_SCHEMA.is_file()
    ver_ok = False
    detail = f"md={'ok' if md_ok else 'MISSING'}; schema={'ok' if sc_ok else 'MISSING'}"
    if sc_ok:
        try:
            sch = json.loads(C.CONTRACT_SCHEMA.read_text())
            ver_ok = sch.get("properties", {}).get("contract_version", {}).get("const") == "1.0"
            detail += f"; contract_version const {'== 1.0' if ver_ok else 'NOT 1.0'}"
        except Exception as e:  # noqa: BLE001
            detail += f"; schema parse error: {e}"
    record("contract_v1.0", md_ok and sc_ok and ver_ok, hard=True, detail=detail)


def check_compute_path_id() -> None:
    try:
        b5 = importlib.import_module("b5_serialize")
        cpid = getattr(b5, "compute_path_id")
        triples = [
            ["MONDO_0005148", "DIS_GEN__ASSOCIATED_WITH", "NCBIGene_5562"],
            ["NCBIGene_5562", "DRG_GEN__TARGET", "CHEMBL_1431"],
        ]
        a, b = cpid(triples), cpid(triples)
        ok = isinstance(a, str) and a.startswith("p_") and a == b
        record("b5_serialize.compute_path_id", ok, hard=True, detail=f"id={a!r} deterministic={a == b}")
    except Exception as e:  # noqa: BLE001
        record("b5_serialize.compute_path_id", False, hard=True, detail=f"import/call error: {e}")


def check_b6_tools() -> None:
    needed = ["link_entity", "retrieve_candidates", "score_pair", "link_and_retrieve"]
    try:
        b6 = importlib.import_module("b6_tools")
        missing = [n for n in needed if not callable(getattr(b6, n, None))]
        record("b6_tools API", not missing, hard=True,
               detail="all present" if not missing else f"missing: {missing}")
    except Exception as e:  # noqa: BLE001
        record("b6_tools API", False, hard=True, detail=f"import error: {e}")


def check_harness() -> None:
    try:
        h = importlib.import_module("d4_harness")
        ev = getattr(h, "evaluate", None)
        ok = callable(ev)
        detail = "evaluate() callable" if ok else "evaluate() missing"
        ranker = getattr(h, "Ranker", None)
        if ranker is not None and hasattr(ranker, "rank"):
            try:
                detail += f"; Ranker.rank{inspect.signature(ranker.rank)}"
            except (TypeError, ValueError):
                detail += "; Ranker.rank signature unavailable"
        else:
            detail += "; no Ranker base exported (see how KGE/mock rankers plug in)"
        record("d4_harness.evaluate", ok, hard=False, detail=detail)
    except Exception as e:  # noqa: BLE001
        record("d4_harness.evaluate", False, hard=False, detail=f"import error: {e}")


def check_whitelist() -> None:
    if not C.WHITELIST_REACHABLE.is_file():
        record("reachable whitelist (1,565)", False, hard=True, detail=f"MISSING @ {C.WHITELIST_REACHABLE}")
        return
    try:
        import polars as pl
        df = pl.read_csv(C.WHITELIST_REACHABLE)
        record("reachable whitelist (1,565)", df.height == C.WHITELIST_REACHABLE_N, hard=True,
               detail=f"{df.height} rows (expect {C.WHITELIST_REACHABLE_N}); cols={df.columns}")
    except Exception as e:  # noqa: BLE001
        record("reachable whitelist (1,565)", False, hard=True, detail=f"read error: {e}")


def check_splits() -> None:
    for split in C.SPLITS:
        hard = split in C.REQUIRED_SPLITS
        model, meta = C.kge_model_pkl(split), C.d1_meta_json(split)
        if not model.is_file() or not meta.is_file():
            record(f"kge assets [{split}]", False, hard=hard,
                   detail=f"model={'ok' if model.is_file() else 'MISSING'}, "
                          f"d1_meta={'ok' if meta.is_file() else 'MISSING'}")
            continue
        try:
            ind = C.read_kge_ind_relation(split)
            hyphen_ok = ind == "DRG-DIS__INDICATION"
            record(f"kge assets [{split}]", hyphen_ok, hard=hard,
                   detail=f"ind_relation={ind!r} (hyphen ok={hyphen_ok}); "
                          f"derived Neo4j TYPE={C.to_neo4j_type(ind)!r}")
        except Exception as e:  # noqa: BLE001
            record(f"kge assets [{split}]", False, hard=hard, detail=f"meta read error: {e}")


def check_reasoner_deps() -> None:
    # transformers present?
    tfm = importlib.util.find_spec("transformers") is not None
    # any guided-JSON backend present?
    backends = {b: importlib.util.find_spec(m) is not None for b, m in
                {"lmfe": "lmformatenforcer", "outlines": "outlines", "xgrammar": "xgrammar"}.items()}
    have_guided = any(backends.values())
    # CUDA visible?
    cuda = False
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        pass
    ok = tfm and have_guided
    detail = (f"transformers={'ok' if tfm else 'MISSING'}; "
              f"guided backends present={[b for b, v in backends.items() if v] or 'NONE — '
              'uv pip install lm-format-enforcer'}; "
              f"selected={C.GUIDED_BACKEND}; cuda={cuda}")
    record("core reasoner deps", ok, hard=False, detail=detail)


def check_neo4j() -> None:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "retracerx")
    try:
        from neo4j import GraphDatabase
    except Exception as e:  # noqa: BLE001
        record("Neo4j filtered/50-types", False, hard=False, detail=f"driver unavailable: {e}")
        return
    try:
        drv = GraphDatabase.driver(uri, auth=(user, pwd))
        with drv.session() as s:
            v = s.run("MATCH (m:_Meta) RETURN m.graph_variant AS v LIMIT 1").single()
            n = s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN count(*) AS n").single()["n"]
        drv.close()
        variant = v["v"] if v else None
        record("Neo4j filtered/50-types", (variant == "filtered") and (n == 50), hard=False,
               detail=f"graph_variant={variant!r}, relationshipTypes={n} (expect filtered/50)")
    except Exception as e:  # noqa: BLE001
        record("Neo4j filtered/50-types", False, hard=False,
               detail=f"connect/query failed (started retracerx-neo4j? sourced neo4j_env.sh?): {e}")


def check_dirs() -> None:
    try:
        C.LOGS.mkdir(parents=True, exist_ok=True)
        C.PHASEC_DIR.mkdir(parents=True, exist_ok=True)
        record("writable logs/ and data/phaseC/", True, hard=True, detail=f"{C.LOGS} , {C.PHASEC_DIR}")
    except Exception as e:  # noqa: BLE001
        record("writable logs/ and data/phaseC/", False, hard=True, detail=str(e))


def main() -> int:
    print(f"Phase C — C0 preflight (CORE)  (repo root: {C.REPO_ROOT})\n")
    check_contract()
    check_compute_path_id()
    check_b6_tools()
    check_harness()
    check_whitelist()
    check_splits()
    check_reasoner_deps()
    check_neo4j()
    check_dirs()

    hard_fail = [r for r in RESULTS if r["hard"] and not r["ok"]]
    soft_fail = [r for r in RESULTS if not r["hard"] and not r["ok"]]

    C.PHASEC_DIR.mkdir(parents=True, exist_ok=True)
    C.C0_PREFLIGHT_JSON.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "core",
        "repo_root": str(C.REPO_ROOT),
        "results": RESULTS,
        "hard_failures": len(hard_fail),
        "soft_warnings": len(soft_fail),
        "gate": "PASS" if not hard_fail else "FAIL",
    }, indent=2))

    print(f"\nWrote {C.C0_PREFLIGHT_JSON}")
    print(f"HARD failures: {len(hard_fail)}   SOFT warnings: {len(soft_fail)}")
    if hard_fail:
        print("C0 GATE: FAIL — resolve the HARD failures above before C1.")
        return 1
    print("C0 GATE: PASS — inputs verified; proceed to C1.")
    if soft_fail:
        print("(soft warnings are non-blocking: e.g. Neo4j not started, or guided lib not yet installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
