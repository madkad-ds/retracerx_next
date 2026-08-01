#!/usr/bin/env bash
# ReTraceRx — Neo4j session preamble for Phase B.
# Usage:  source scripts/neo4j_env.sh
# Starts the container (idempotent), waits for Bolt, exports NEO4J_* for the B3
# scripts, and sanity-pings APOC + the leakage-variant marker.

# --- connection facts (Phase A §2) ---
export NEO4J_CONTAINER="${NEO4J_CONTAINER:-retracerx-neo4j}"
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"   # bolt://, NOT neo4j://
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-retracerx}"
export NEO4J_DB="${NEO4J_DB:-neo4j}"

# --- start container if not running (no restart policy is set; A0 note) ---
if [ -z "$(docker ps -q -f name="^${NEO4J_CONTAINER}$")" ]; then
  echo "starting ${NEO4J_CONTAINER} ..."
  docker start "${NEO4J_CONTAINER}" >/dev/null
fi
# NB: B3 is READ-ONLY — do NOT chown neo4j/import here (that's only for emit/import
# cycles). Never chown neo4j/data.

# --- wait for Bolt to accept a query (docker start returns before DB is ready) ---
python - <<'PY'
import os, sys, time
from neo4j import GraphDatabase
uri  = os.environ["NEO4J_URI"]; auth = (os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
db   = os.environ["NEO4J_DB"]
for i in range(60):
    try:
        drv = GraphDatabase.driver(uri, auth=auth)
        with drv.session(database=db) as s:
            s.run("RETURN 1").single()
        drv.close(); print("neo4j ready"); sys.exit(0)
    except Exception:
        time.sleep(2)
print("neo4j did NOT become ready in ~120s", file=sys.stderr); sys.exit(1)
PY

# --- sanity ping: APOC present, leakage variant, 50 surviving types ---
python - <<'PY'
import os
from neo4j import GraphDatabase
drv = GraphDatabase.driver(os.environ["NEO4J_URI"],
                           auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
with drv.session(database=os.environ["NEO4J_DB"]) as s:
    try:
        apoc = s.run("RETURN apoc.version() AS v").single()["v"]
    except Exception as e:
        apoc = f"MISSING ({e})"
    ntypes = len(s.run("CALL db.relationshipTypes() YIELD relationshipType "
                       "RETURN relationshipType").value())
    m = s.run("MATCH (m:_Meta) RETURN properties(m) AS p").single()
    variant = (m["p"] if m else None)
drv.close()
print(f"APOC: {apoc}")
print(f"relationship types: {ntypes} (expect 50 = filtered variant)")
print(f"variant marker: {variant}")
if "MISSING" in str(apoc):
    print("!!! APOC missing — recreate container with -e NEO4J_PLUGINS='[\"apoc\"]' "
          "(data volume persists; password not reset).")
if ntypes != 50:
    print(f"!!! {ntypes} types, expected 50 — you may be on the FULL (unfiltered) store; "
          "B3 would leak direct drug-disease edges. Check A3 Step 5.")
PY
