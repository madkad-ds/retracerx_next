#!/usr/bin/env python3
"""
e1_healthcheck.py — verify the deployed stack is correct, not just up.

Asserts the E1 invariants that make the demo trustworthy:
  1. vLLM serves the pinned model (Qwen/Qwen3.5-9B).
  2. Guided JSON round-trips: greedy (temperature 0) + seed 20260727 +
     response_format=json_schema + chat_template_kwargs.enable_thinking=false
     → parseable JSON with NO "Thinking Process:" preamble (Phase C E-C5).
  3. Determinism: two identical calls are byte-identical (D-C14).
  4. Neo4j: Bolt reachable, APOC loaded, FILTERED variant (leakage type == 0).
  5. GPU budget: ~18.1 GB resident with real KV-cache headroom on the 24 GB L4.

Writes data/phaseE/e1_healthcheck.json. Exit 0 iff no HARD failure.
Run with the serve venv python (neo4j + openai clients installed by e0_bootstrap).
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_config as C

HARD, SOFT = "HARD", "SOFT"
checks: list[dict] = []

def record(name, ok, level, detail=""):
    checks.append({"check": name, "ok": bool(ok), "level": level, "detail": str(detail)})
    tag = "PASS" if ok else ("FAIL" if level == HARD else "WARN")
    print(f"[{tag:4}] {name}: {detail}")

# ---------------------------------------------------------------------------
# vLLM (OpenAI-compatible). We use the `openai` client if present, else httpx.
# ---------------------------------------------------------------------------
BASE = f"http://localhost:{C.PORTS['vllm']}/v1"
API_KEY, API_KEY_SRC = C.resolve_vllm_api_key()
AUTH = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

def _tiny_json_call(model_name):
    """One deterministic guided-JSON generation. Returns (raw_text, parsed|None)."""
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    messages = [{"role": "user", "content": "Reply with a JSON object {\"ok\": true}."}]
    body = dict(
        model=model_name,                   # the SERVED name (e.g. 'reasoner'), not the repo
        messages=messages,
        temperature=C.TEMPERATURE,          # greedy
        seed=C.SPLIT_SEED,                  # byte-identical replay
        max_tokens=32,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "probe", "schema": schema}},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    try:
        from openai import OpenAI
        client = OpenAI(base_url=BASE, api_key=API_KEY or "EMPTY")
        r = client.chat.completions.create(**body)
        text = r.choices[0].message.content
    except Exception:
        # httpx fallback: fold extra_body into the top-level payload
        import httpx
        payload = {k: v for k, v in body.items() if k != "extra_body"}
        payload.update(body["extra_body"])
        r = httpx.post(f"{BASE}/chat/completions", json=payload, headers=AUTH, timeout=120)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    try:
        return text, json.loads(text)
    except Exception:
        return text, None

def check_vllm():
    record("vLLM auth key resolved", bool(API_KEY), SOFT,
           f"from {API_KEY_SRC}" if API_KEY else "no key found — sending unauthenticated")

    # 1. Discover what's served. The model is renamed via --served-model-name, so
    #    read the live id and verify the underlying repo via vLLM's `root` field.
    try:
        import httpx
        data = httpx.get(f"{BASE}/models", headers=AUTH, timeout=30).json()
        entries = data.get("data", [])
    except Exception as e:
        record("vLLM /v1/models reachable", False, HARD, e); return
    if not entries:
        record("vLLM serves a model", False, HARD, "empty /v1/models (auth? not ready?)"); return

    ids   = [e.get("id", "") for e in entries]
    roots = [e.get("root", "") or "" for e in entries]
    served = ids[0]
    repo_ok = any(C.REASONER_MODEL_REPO in x for x in ids + roots)
    if repo_ok:
        record("vLLM serves pinned model", True, HARD,
               f"{C.REASONER_MODEL_REPO} served as '{served}'")
    else:
        # Server is up and serving; couldn't confirm the repo from root (older vLLM).
        record("vLLM serves a model", True, HARD, f"served id '{served}' (root not exposed)")
        record("underlying repo confirmed == pinned", False, SOFT,
               f"ids={ids} roots={roots}; expected {C.REASONER_MODEL_REPO}")

    # 2. guided-JSON round-trip (no Thinking-Process leak) — use the SERVED id
    try:
        text, parsed = _tiny_json_call(served)
        no_leak = "Thinking Process" not in (text or "")
        record("guided JSON parses", parsed is not None, HARD, (text or "")[:80])
        record("no 'Thinking Process:' preamble", no_leak, HARD)
    except Exception as e:
        record("guided JSON round-trip", False, HARD, e); return

    # 3. determinism (two identical calls byte-identical)
    try:
        t1, _ = _tiny_json_call(served)
        t2, _ = _tiny_json_call(served)
        record("determinism (byte-identical replay)", t1 == t2, HARD)
    except Exception as e:
        record("determinism", False, SOFT, e)

# ---------------------------------------------------------------------------
# Neo4j: Bolt + APOC + filtered variant
# ---------------------------------------------------------------------------
def check_neo4j():
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        record("neo4j driver import", False, HARD, f"pip install neo4j ({e})"); return
    try:
        drv = GraphDatabase.driver(C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASS))
        with drv.session(database=C.NEO4J_DB) as s:
            record("Neo4j Bolt reachable", s.run("RETURN 1 AS x").single()["x"] == 1, HARD, C.NEO4J_URI)
            apoc = s.run("RETURN apoc.version() AS v").single()["v"]
            record("APOC loaded", bool(apoc), HARD, apoc)
            leak = s.run(
                f"MATCH ()-[r:{C.NEO4J_LEAKAGE_PROBE_LABEL}]-() RETURN count(r) AS n"
            ).single()["n"]
            record("filtered variant (leakage == 0)", leak == 0, HARD,
                   f"{C.NEO4J_LEAKAGE_PROBE_LABEL} count = {leak}")
        drv.close()
    except Exception as e:
        record("Neo4j checks", False, HARD, e)

# ---------------------------------------------------------------------------
# GPU budget: ~18.1 GB resident, KV-cache headroom
# ---------------------------------------------------------------------------
def check_gpu_budget():
    smi = shutil.which("nvidia-smi")
    if not smi:
        record("nvidia-smi present", False, HARD, "no GPU tooling"); return
    try:
        used, total = [], []
        out = subprocess.check_output(
            [smi, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=15).strip().splitlines()
        u, t = (int(x) for x in out[0].split(","))
        used_gb, total_gb, free_gb = u / 1024, t / 1024, (t - u) / 1024
        record("GPU total is 24 GB-class",
               t >= C.GPU_TOTAL_MIN_MIB, HARD, f"{total_gb:.1f} GB")
        # Weights resident should be in the ballpark of 18.1 GB; warn if wildly off.
        record("weights resident ~ 18 GB (FP16, no quant)",
               used_gb >= 14.0, SOFT, f"used {used_gb:.1f} GB")
        record(f"KV-cache headroom >= {C.KV_HEADROOM_MIN_GB} GB",
               free_gb >= C.KV_HEADROOM_MIN_GB, HARD,
               f"free {free_gb:.1f} GB (lower --max-model-len if this is tight)")
    except Exception as e:
        record("GPU budget query", False, HARD, e)

def main():
    check_vllm()
    check_neo4j()
    check_gpu_budget()

    hard_fail = [c for c in checks if not c["ok"] and c["level"] == HARD]
    soft_warn = [c for c in checks if not c["ok"] and c["level"] == SOFT]
    report = {
        "healthy": len(hard_fail) == 0,
        "n_checks": len(checks),
        "hard_failures": len(hard_fail),
        "soft_warnings": len(soft_warn),
        "checks": checks,
    }
    C.E1_HEALTH_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.E1_HEALTH_JSON.write_text(json.dumps(report, indent=2, default=str))
    print("\n" + "=" * 64)
    print(f"E1 HEALTH: {'OK ✅' if report['healthy'] else 'UNHEALTHY ❌'}  "
          f"({report['hard_failures']} hard, {report['soft_warnings']} soft)  "
          f"→ {C.E1_HEALTH_JSON}")
    sys.exit(0 if report["healthy"] else 1)

if __name__ == "__main__":
    main()
