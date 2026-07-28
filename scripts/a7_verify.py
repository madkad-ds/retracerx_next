#!/usr/bin/env python3
"""
A7 verification gate. Exit 1 => do NOT proceed to A3 Step 5 / A8.

Gates:
  G1  removed_edge_ids.csv carries endpoint_key; height 79,949; DRG-DIS = 70,380 (whole label).
  G2  removed set is exactly the 7 leakage (label,relation) pairs.
  G3  endpoint_key on undirected DRG-DIS rows contains the U+241F separator (copied, not rebuilt).
  G4  holdout positives keyed on (okg_drug_id, okg_disease_id); unique; no nulls.
  G5  disease collapse applied: no holdout disease id is a collapsed ALTERNATE (all -> chosen).
  G6  holdout INTERSECT known_indications_minus_holdout == empty (D19).
  G7  candidate_whitelist = 1,700; no LCC-excluded drug id present.
  G8  no OBA disease id appears in the holdout disease space.
  G9  MANIFEST carries dataset_version, graph_variant, split_seed + sha256 for every output.
  G10 positives and negatives are disjoint (Approved-wins).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import polars as pl

OUT   = Path("data/holdout")
EXCL  = OUT / "inference_exclusions"
DIS_COLLAPSE = "data/manifest/a6_disease_collapse.parquet"
LCC_EXCL     = "data/manifest/a3_lcc_excluded.parquet"
US = "\u241f"

EXPECT_REMOVED, EXPECT_DRGDIS, EXPECT_WHITELIST = 79_949, 70_380, 1_700
LEAK_PAIRS = {
    ("DRG-DIS", "INDICATION"), ("DRG-DIS", "CONTRAINDICATION"), ("DRG-DIS", "OFF_LABEL_USE"),
    ("DRG-PHE", "CONTRAINDICATION"), ("DRG-PHE", "INDICATION"), ("DRG-PHE", "OFF_LABEL_USE"),
    ("DRG-BPO", "INDICATION"),
}

fails = []
def check(cond: bool, gate: str, msg: str):
    print(f"  {'[PASS]' if cond else '[FAIL]'} {gate}: {msg}")
    if not cond:
        fails.append(gate)

def nlab(s: str) -> str:
    return s.replace("_", "-")

# ---- G1 / G2 / G3 : removed_edge_ids.csv --------------------------------------
rm = pl.read_csv(OUT / "removed_edge_ids.csv")
check({"from", "to", "label", "relation", "endpoint_key"}.issubset(rm.columns),
      "G1a", f"columns include endpoint_key -> {rm.columns}")
check(rm.height == EXPECT_REMOVED, "G1b", f"{rm.height:,} == {EXPECT_REMOVED:,}")
drgdis = rm.filter(pl.col("label").map_elements(nlab, return_dtype=pl.Utf8) == "DRG-DIS").height
check(drgdis == EXPECT_DRGDIS, "G1c", f"DRG-DIS {drgdis:,} == {EXPECT_DRGDIS:,} (whole label, E3)")

pairs = {(nlab(l), r) for l, r in rm.select(["label", "relation"]).unique().iter_rows()}
check(pairs == LEAK_PAIRS, "G2", f"exactly the 7 leakage pairs -> {sorted(pairs)}")

und = rm.filter(pl.col("label").map_elements(nlab, return_dtype=pl.Utf8) == "DRG-DIS")
check(und["endpoint_key"].str.contains(US, literal=True).all(),
      "G3", "endpoint_key on DRG-DIS carries U+241F (copied, not rebuilt)")

# ---- G4 : holdout key ---------------------------------------------------------
hp = pl.read_csv(OUT / "holdout_positives.csv")
check({"okg_drug_id", "okg_disease_id"}.issubset(hp.columns),
      "G4a", f"keyed on node ids -> {hp.columns}")
check(hp.select(["okg_drug_id", "okg_disease_id"]).n_unique() == hp.height,
      "G4b", "(okg_drug_id, okg_disease_id) unique")
check(hp.filter(pl.col("okg_drug_id").is_null() | pl.col("okg_disease_id").is_null()).height == 0,
      "G4c", "no null keys")

# ---- G5 : disease collapse applied -------------------------------------------
dc = pl.read_parquet(DIS_COLLAPSE)
alt_col = "all_okg_dis_ids" if "all_okg_dis_ids" in dc.columns else dc.columns[-1]
chosen  = set(dc["chosen_okg_dis_id"].to_list())
alternates = set(dc.select(alt_col).explode(alt_col).to_series().to_list()) - chosen
leaked_alt = set(hp["okg_disease_id"].to_list()) & alternates
check(len(leaked_alt) == 0, "G5",
      f"no collapsed-alternate disease id in holdout ({len(leaked_alt)} leaked)")

# ---- G6 : D19 exclusion is holdout-safe --------------------------------------
km = pl.read_csv(EXCL / "known_indications_minus_holdout.csv")
hp_pairs = set(zip(hp["okg_drug_id"].to_list(), hp["okg_disease_id"].to_list()))
km_pairs = set(zip(km["okg_drug_id"].to_list(), km["okg_disease_id"].to_list()))
check(len(hp_pairs & km_pairs) == 0, "G6",
      f"holdout not inside known-minus-holdout ({len(hp_pairs & km_pairs)} overlap)")

# ---- G7 : whitelist -----------------------------------------------------------
wl = pl.read_csv(OUT / "candidate_whitelist.csv")
check(wl.height == EXPECT_WHITELIST, "G7a", f"whitelist {wl.height:,} == {EXPECT_WHITELIST:,}")
lcc = pl.read_parquet(LCC_EXCL)
lcc_drugs = set(lcc.filter(pl.col("label") == "DRG")["id"].to_list())
wl_col = "okg_drug_id" if "okg_drug_id" in wl.columns else wl.columns[0]
check(len(set(wl[wl_col].to_list()) & lcc_drugs) == 0, "G7b",
      "no LCC-excluded drug in whitelist")

# ---- G8 : no OBA in holdout ---------------------------------------------------
oba_in_holdout = hp.filter(pl.col("okg_disease_id").str.starts_with("OBA")).height
check(oba_in_holdout == 0, "G8", f"no OBA disease in holdout ({oba_in_holdout})")

# ---- G9 : MANIFEST ------------------------------------------------------------
man = json.loads((OUT / "MANIFEST.json").read_text())
has_keys = all(k in man for k in ("dataset_version", "graph_variant", "split_seed"))
outs_ok = all("sha256" in v for v in man.get("outputs", {}).values())
check(has_keys and outs_ok, "G9",
      "MANIFEST carries dataset_version + graph_variant + split_seed + per-output sha256")

# ---- G10 : positives / negatives disjoint ------------------------------------
hn = pl.read_csv(OUT / "holdout_negatives.csv")
hn_pairs = set(zip(hn["okg_drug_id"].to_list(), hn["okg_disease_id"].to_list()))
check(len(hp_pairs & hn_pairs) == 0, "G10", f"holdout pos/neg disjoint ({len(hp_pairs & hn_pairs)})")

print()
if fails:
    print(f"[A7 GATE] FAILED: {fails} - do NOT proceed.")
    sys.exit(1)
print("[A7 GATE] 10/10 PASS - proceed to A3 Step 5, then A8.")
