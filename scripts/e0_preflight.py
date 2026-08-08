#!/usr/bin/env python3
"""
e0_preflight.py — Phase E "Definition of Done" input gate.

Verifies every artifact E1..E9 consume from the FROZEN upstream phases exist and
match their pins, BEFORE anything is deployed. Same spirit as Phase C's
c0_preflight: 0 hard failures = gate open. Writes data/phaseE/e0_preflight.json.

Design rules:
  * stdlib-only (no pandas/torch) so it runs before the serve venv exists.
  * NEVER unpickle or load models — existence + shape checks only.
  * Live-service checks (Bolt reachable, filtered-variant assert, vLLM up) belong
    to E1 (e1_healthcheck.py), because the containers are started in E1. Here we
    only require the *ingredients* to be present (neo4j_env.sh, plugins bind with
    an APOC jar, the pinned digest, the FP16 serve script, GPU present).

Exit code 0 iff no HARD failure.
"""
from __future__ import annotations
import csv, json, os, re, shutil, socket, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_config as C

HARD, SOFT = "HARD", "SOFT"
checks: list[dict] = []

def record(name, ok, level, detail=""):
    checks.append({"check": name, "ok": bool(ok), "level": level, "detail": str(detail)})
    tag = "PASS" if ok else ("FAIL" if level == HARD else "WARN")
    print(f"[{tag:4}] {name}: {detail}")

def exists(name, path: Path, level=HARD):
    record(name, path.exists(), level, path)

# -- 1. Retrieval contract v1.0 (Phase B D-B1) --------------------------------
def check_contract():
    exists("contract.md", C.CONTRACT_MD)
    exists("contract.schema.json", C.CONTRACT_SCHEMA)
    if C.CONTRACT_MD.exists():
        txt = C.CONTRACT_MD.read_text(errors="ignore")
        ok = ("1.0" in txt) and ("contract_version" in txt or "contract version" in txt.lower())
        record("contract.version==1.0", ok, HARD, "contract_version 1.0 referenced")

# -- 2. Shared serializer + tools importable (D-B18 / D-B21) ------------------
def check_python_imports():
    # We check the *files* exist here (import needs the serve venv). e1 imports live.
    exists("b5_serialize.py (compute_path_id)", C.B5_SERIALIZE)
    exists("b6_tools.py", C.B6_TOOLS)
    for fn in ("compute_path_id",):
        ok = C.B5_SERIALIZE.exists() and re.search(rf"def\s+{fn}\b",
                                                    C.B5_SERIALIZE.read_text(errors="ignore"))
        record(f"b5_serialize defines {fn}", bool(ok), HARD)
    for fn in ("link_entity", "retrieve_candidates", "score_pair", "link_and_retrieve"):
        ok = C.B6_TOOLS.exists() and re.search(rf"def\s+{fn}\b",
                                               C.B6_TOOLS.read_text(errors="ignore"))
        record(f"b6_tools defines {fn}", bool(ok), HARD)

# -- 3. Whitelist = 1,565 with inchi_key (D-C10 recovery key) -----------------
def check_whitelist():
    if not C.WHITELIST_CSV.exists():
        record("whitelist csv", False, HARD, C.WHITELIST_CSV); return
    with C.WHITELIST_CSV.open(newline="") as f:
        r = csv.reader(f); header = next(r, [])
        n = sum(1 for _ in r)
    record("whitelist has inchi_key col", "inchi_key" in header, HARD, header)
    record(f"whitelist rows == {C.WHITELIST_REACHABLE_N}",
           n == C.WHITELIST_REACHABLE_N, HARD, f"found {n}")

# -- 4. Per-split RotatE + meta + frozen positives (D-B14 / splits) -----------
def check_splits():
    for s in C.SPLITS:
        exists(f"kge model [{s}]", C.kge_model(s))
        exists(f"kge meta  [{s}]", C.kge_meta(s))
        exists(f"holdout positives [{s}]", C.split_positives(s))

# -- 5. Neo4j ingredients (Phase A/B) -----------------------------------------
def _container_running(name: str) -> bool:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name=^{name}$", "--filter", "status=running", "-q"],
            text=True, timeout=10).strip()
        return bool(out)
    except Exception:
        return False

def _apoc_live_ok() -> bool:
    try:
        out = subprocess.run(
            ["docker", "exec", C.NEO4J_CONTAINER, "cypher-shell",
             "-u", C.NEO4J_USER, "-p", C.NEO4J_PASS, "--format", "plain",
             "RETURN apoc.version()"],
            capture_output=True, text=True, timeout=30)
        return out.returncode == 0 and any(ch.isdigit() for ch in out.stdout)
    except Exception:
        return False

def check_neo4j_ingredients():
    exists("neo4j_env.sh", C.NEO4J_ENV_SH)
    plugins = C.REPO / "neo4j" / "plugins"
    data    = C.REPO / "neo4j" / "data"
    record("neo4j/plugins bind exists", plugins.is_dir(), HARD, plugins)
    record("neo4j/data bind exists", data.is_dir(), HARD, data)

    # APOC: authoritative check is LIVE (apoc.version()); the on-disk jar is a
    # best-effort fallback. Recursive + case-insensitive so we don't miss e.g.
    # apoc-core.jar in a subdir (Phase B E-B5).
    jars = [p for p in (plugins.rglob("*.jar") if plugins.is_dir() else [])
            if "apoc" in p.name.lower()]
    if _container_running(C.NEO4J_CONTAINER):
        if _apoc_live_ok():
            record("APOC loaded (live apoc.version())", True, HARD, "container up")
        else:
            record("APOC loaded (live apoc.version())", False, HARD,
                   "container up but apoc.version() failed — APOC not installed/loaded")
    elif jars:
        record("APOC jar in plugins bind", True, HARD, jars[0])
    else:
        record("APOC present", False, SOFT,
               "container down and no apoc*.jar on the bind — E1 healthcheck confirms live")

# -- 6. vLLM serving artifacts to reuse: pinned digest + FP16 script (E3) ------
def check_vllm_pins():
    exists("c1_serve_vllm_fp16.sh", C.FP16_SERVE_SH)
    img, src = C.resolve_vllm_image()
    record("vLLM image is DIGEST-pinned (sha256)", bool(img), HARD,
           f"{img[:36]}…  (from {src})" if img else
           "no sha256 digest found in vllm.env or c1_serve_vllm_fp16.sh")

# -- 7. GPU present with 24 GB-class memory (E1) ------------------------------
def check_gpu():
    smi = shutil.which("nvidia-smi")
    if not smi:
        record("nvidia-smi present", False, HARD, "GPU driver / nvidia-smi not found"); return
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip().splitlines()
        total = max(int(x) for x in out)
        record(f"GPU total >= {C.GPU_TOTAL_MIN_MIB} MiB", total >= C.GPU_TOTAL_MIN_MIB,
               HARD, f"{total} MiB")
    except Exception as e:
        record("GPU query", False, HARD, e)

# -- 8. Dashboard inputs for E5 (Phase D D6/D7 + Phase C C3/C8) ---------------
def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def check_dashboard_inputs():
    # Per-cell recovery CIs (D-D21). Documented at artifacts/d7_stats/ but the
    # repo layout may differ; search candidates, else fall back to paired CIs.
    percell = _first_existing([
        C.D7_PERCELL_CI,
        C.REPO / "data" / "phaseD" / "d7_percell_ci.csv",
        C.REPO / "data" / "phaseC" / "d7_percell_ci.csv",
        C.REPO / "artifacts" / "d7_percell_ci.csv",
        C.REPO / "data" / "phaseC" / "paired" / "d7_percell_ci.csv",
    ])
    paired_ok = C.PAIRED_SUMMARY.exists()
    if percell is not None:
        record("d7_percell_ci.csv (per-cell CIs)", True, HARD, percell)
    elif paired_ok:
        record("d7_percell_ci.csv (per-cell CIs)", False, SOFT,
               "not found; dashboard will back CIs with paired_summary.csv instead")
    else:
        record("d7 CIs (per-cell or paired)", False, HARD,
               "neither d7_percell_ci.csv nor paired_summary.csv found")
    record("paired_summary.csv", paired_ok, HARD, C.PAIRED_SUMMARY)
    for s in C.SPLITS:
        exists(f"c3_faithfulness [{s}]", C.c3_faithfulness(s))
        exists(f"c8_calibration  [{s}]", C.c8_calibration(s))

# -- 9. Host tooling: uv + ports free (soft; bootstrap/up handle these) -------
def check_host_tooling():
    record("uv on PATH", bool(shutil.which("uv")), SOFT, "needed for the serve venv")
    record("docker on PATH", bool(shutil.which("docker")), HARD, "needed for E1")
    for name, port in C.PORTS.items():
        free = _port_free(port)
        record(f"port {port} ({name}) free", free, SOFT,
               "in use — likely the Phase C neo4j/vllm still up (fine for E1)" if not free else "")

def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) != 0

def main():
    check_contract()
    check_python_imports()
    check_whitelist()
    check_splits()
    check_neo4j_ingredients()
    check_vllm_pins()
    check_gpu()
    check_dashboard_inputs()
    check_host_tooling()

    hard_fail = [c for c in checks if not c["ok"] and c["level"] == HARD]
    soft_warn = [c for c in checks if not c["ok"] and c["level"] == SOFT]
    report = {
        "gate_open": len(hard_fail) == 0,
        "n_checks": len(checks),
        "hard_failures": len(hard_fail),
        "soft_warnings": len(soft_warn),
        "checks": checks,
    }
    C.E0_PREFLIGHT_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.E0_PREFLIGHT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print("\n" + "=" * 64)
    print(f"E0 GATE: {'OPEN ✅' if report['gate_open'] else 'BLOCKED ❌'}  "
          f"({report['hard_failures']} hard, {report['soft_warnings']} soft)  "
          f"→ {C.E0_PREFLIGHT_JSON}")
    sys.exit(0 if report["gate_open"] else 1)

if __name__ == "__main__":
    main()
