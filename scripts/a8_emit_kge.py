#!/usr/bin/env python3
"""A8: emit leakage-filtered KGE triples from the filtered canon_edges.
Decisions (frozen in a8_decisions.json):
  D-A8-1 relation granularity = composite  (label__relation)
  D-A8-2 undirected orientation = single   (inverses handled in PyKEEN, Phase D)
  D-A8-3 self-loops            = drop       (baseline hygiene; kept in the graph)
"""
import json, hashlib, sys
from pathlib import Path
import polars as pl

CANON_EDGES = "data/canon/canon_edges/*.parquet"
REMOVED     = "data/holdout/removed_edge_ids.csv"
OUT_TSV     = Path("data/kge_triples.tsv")
DEC_JSON    = Path("data/manifest/a8_decisions.json")
MAN_JSON    = Path("data/manifest/a8_manifest.json")

# --- hard expectations (measured upstream — A3 Step 5 gate / census) ---
EXPECT_FILTERED   = 21_733_867     # rows after anti-join, BEFORE the self-loop drop
EXPECT_COMPOSITE  = 50             # distinct label__relation after removing 7 leakage pairs
EXPECT_SELFLOOPS  = 5_761          # A2 M-series; sanity only

# --- decisions ---
REL_GRANULARITY = "composite"      # D-A8-1
UNDIRECTED      = "single"         # D-A8-2  ("single" | "both")
SELF_LOOPS      = "drop"           # D-A8-3  ("drop"   | "keep")

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def die(msg: str):
    sys.exit(f"[A8 FATAL] {msg}")

# 1) removed set — READ endpoint_key VERBATIM (U+241F). Never rebuild it.
#    Anti-join key is (endpoint_key, relation): endpoint_key already encodes both node
#    identities (hence the label pair), so this cannot over- or under-remove.
removed = pl.read_csv(REMOVED)
for col in ("endpoint_key", "relation"):
    if col not in removed.columns:
        die(f"removed_edge_ids.csv missing '{col}' — A7 contract violated")
if removed.height != 79_949:
    die(f"removed rows {removed.height:,} != 79,949 — wrong holdout bundle")
removed_key = removed.select(["endpoint_key", "relation"]).unique().lazy()

# 2) filtered edge stream (lazy — never materialize 21.8M rows)
edges = pl.scan_parquet(CANON_EDGES)
filtered = edges.join(removed_key, on=["endpoint_key", "relation"], how="anti")

# GATE A — the join proof. Must reproduce the A3 Step 5 measured count EXACTLY.
n_filtered = filtered.select(pl.len()).collect().item()
if n_filtered != EXPECT_FILTERED:
    die(f"filtered rows {n_filtered:,} != {EXPECT_FILTERED:,} "
        f"(anti-join drifted — check the U+241F separator was preserved on read)")
print(f"[A8] GATE A ok — filtered edges = {n_filtered:,}")

# 3) build the composite relation (D-A8-1) and count self-loops (D-A8-3)
work = filtered.with_columns(
    relation_out = pl.concat_str([pl.col("label"), pl.col("relation")], separator="__"),
    is_self_loop = (pl.col("from") == pl.col("to")),
)

# GATE B — composite relation cardinality must be 50 (== A3 Step 5 surviving types).
n_types = work.select(pl.col("relation_out").n_unique()).collect().item()
if n_types != EXPECT_COMPOSITE:
    die(f"composite relations {n_types} != {EXPECT_COMPOSITE} "
        f"(bare-relation conflation, or a label/relation drifted)")
print(f"[A8] GATE B ok — composite relations = {n_types}")

# self-loop count on the FILTERED set (sanity — should equal 5,761 since none are leakage)
n_self = work.filter(pl.col("is_self_loop")).select(pl.len()).collect().item()
if n_self != EXPECT_SELFLOOPS:
    print(f"[A8] WARN self-loops = {n_self:,} (expected {EXPECT_SELFLOOPS:,}); "
          f"inspect their label__relation before proceeding")

# 4) apply the self-loop decision (record BOTH sides of the count)
if SELF_LOOPS == "drop":
    work = work.filter(~pl.col("is_self_loop"))
n_after_self = work.select(pl.len()).collect().item()
print(f"[A8] self-loops {SELF_LOOPS} — rows {n_filtered:,} -> {n_after_self:,} "
      f"(delta {n_filtered - n_after_self:,})")

# 5) apply the undirected decision
if UNDIRECTED == "both":
    # duplicate undirected, non-self-loop edges with swapped endpoints
    undirected_dupes = (work.filter(pl.col("undirected") & ~pl.col("is_self_loop"))
                            .select(from_=pl.col("to"), to=pl.col("from"),
                                    relation_out=pl.col("relation_out")))
    triples = (work.select(from_=pl.col("from"), to=pl.col("to"),
                           relation_out=pl.col("relation_out"))
                   .select(["from_", "relation_out", "to"]))  # (h, r, t)
    triples = pl.concat([triples, undirected_dupes.select(["from_", "relation_out", "to"])])
else:  # "single" (recommended)
    triples = work.select(["from", "relation_out", "to"])     # (h, r, t)

n_triples = triples.select(pl.len()).collect().item()

# 6) sink to TSV — no header, tab-separated, exactly the PyKEEN default shape
OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
triples.sink_csv(OUT_TSV, separator="\t", include_header=False)
print(f"[A8] wrote {OUT_TSV} — {n_triples:,} triples")

# 7) freeze decisions + manifest (sha256 both sides of every count)
DEC_JSON.parent.mkdir(parents=True, exist_ok=True)
DEC_JSON.write_text(json.dumps({
    "stage": "A8",
    "relation_granularity": REL_GRANULARITY,
    "undirected_orientation": UNDIRECTED,
    "self_loops": SELF_LOOPS,
    "counts": {
        "filtered_join_proof": n_filtered,       # 21,733,867
        "self_loops_on_filtered": n_self,        # 5,761
        "after_self_loop_decision": n_after_self,
        "emitted_triples": n_triples,
    },
    "phaseD_note": "undirected symmetry handled via PyKEEN create_inverse_triples=True "
                   "(single-orientation on disk).",
}, indent=2))
MAN_JSON.write_text(json.dumps({
    "output": str(OUT_TSV),
    "sha256": sha256(OUT_TSV),
    "rows": n_triples,
    "inputs": {"canon_edges_glob": CANON_EDGES, "removed": REMOVED,
               "removed_sha256": sha256(Path(REMOVED))},
}, indent=2))
print("[A8] a8_decisions.json + a8_manifest.json written")