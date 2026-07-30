#!/usr/bin/env python3
"""A8 gates. Exit 1 => do not proceed to Phase D."""
import json, sys
from pathlib import Path
import polars as pl

TSV = "data/kge_triples.tsv"
dec = json.loads(Path("data/manifest/a8_decisions.json").read_text())

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond

lf = pl.scan_csv(TSV, separator="\t", has_header=False,
                 new_columns=["h", "r", "t"])
n     = lf.select(pl.len()).collect().item()
nrel  = lf.select(pl.col("r").n_unique()).collect().item()
nloop = lf.filter(pl.col("h") == pl.col("t")).select(pl.len()).collect().item()
ndrgdis = lf.filter(pl.col("r").str.starts_with("DRG_DIS__")).select(pl.len()).collect().item()

ok = True
# G1 — row count matches the frozen decision
ok &= check("G1 row count", n == dec["counts"]["emitted_triples"],
            f"{n:,} == {dec['counts']['emitted_triples']:,}")
# G2 — exactly 50 composite relations survive (single-orientation) — the A3 Step 5 tie-in
if dec["undirected_orientation"] == "single":
    ok &= check("G2 composite relations", nrel == 50, f"{nrel} == 50")
# G3 — leakage is GONE: no DRG_DIS__* relation may appear at all (E3)
ok &= check("G3 no DRG-DIS leakage", ndrgdis == 0, f"DRG_DIS__* rows = {ndrgdis}")
# G4 — self-loop decision honoured
if dec["self_loops"] == "drop":
    ok &= check("G4 self-loops dropped", nloop == 0, f"self-loops = {nloop}")
else:
    ok &= check("G4 self-loops kept", nloop == dec["counts"]["self_loops_on_filtered"])
# G5 — every relation is composite (contains '__'); none is a bare code
nbare = lf.filter(~pl.col("r").str.contains("__")).select(pl.len()).collect().item()
ok &= check("G5 all relations composite", nbare == 0, f"bare-relation rows = {nbare}")

sys.exit(0 if ok else 1)