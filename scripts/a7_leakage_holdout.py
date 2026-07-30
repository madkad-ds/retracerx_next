#!/usr/bin/env python3
"""
A7 - Leakage control & freeze the holdout.

Produces data/holdout/:
  removed_edge_ids.csv    (from,to,label,relation,endpoint_key)  <- endpoint_key REQUIRED
  holdout_positives.csv   (okg_drug_id, okg_disease_id) + inchikey/drugbank annotations
  holdout_negatives.csv   same keys, from repoDB Terminated/Withdrawn/Suspended
  candidate_whitelist.csv Broad Launched INTERSECT OKG, minus LCC-excluded
  drug_xwalk.csv / disease_xwalk.csv
  splits/{random,drug_cold,disease_cold}_holdout_positives.csv (+negatives)
  inference_exclusions/{known_indications_minus_holdout,oba_disease_ids,lcc_excluded_drug_ids}.csv
  a7_decisions.json
  MANIFEST.json

Honors (STATUS(3) contracts):
  - E3          : DRG-DIS is entirely leakage -> delete the whole label.
  - CONTRA rule : DRG-DIS whole label; DRG-PHE removed in full (see a7_decisions / DECISION below).
  - endpoint_key: copied verbatim from canon_edges (U+241F separator), never rebuilt.
  - A6 open r.3 : apply a6_disease_collapse (umls_cui -> chosen node) BEFORE keying.
  - key         : (okg_drug_id, okg_disease_id), NOT (inchikey, ...)  (E15/E18/984-keyless).
  - E16         : LCC-excluded drugs excluded by ARTIFACT (a3_lcc_excluded), never by degree.
  - E19         : OBA excluded via NOT id STARTS WITH 'OBA', never id_prefix <> 'OBA'.
  - D19         : inference exclusion = known indications MINUS the holdout set.
  - D21/E7      : negatives = Terminated + Withdrawn + Suspended.
  - temporal    : dropped (2.0 is a property-only refresh).

Run from repo root:  python scripts/a7_leakage_holdout.py
"""
from __future__ import annotations
import json, hashlib, sys
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl

# ----------------------------------------------------------------------------- paths
CANON_EDGES  = "data/canon/canon_edges/*.parquet"
CANON_NODES  = "data/canon/canon_nodes.parquet"
GOLD_POS     = "data/gold/gold_positives.parquet"
GOLD_NEG     = "data/gold/gold_negatives.parquet"
REPODB_RES   = "data/gold/repodb_resolved.parquet"
DRUG_XWALK   = "data/holdout/drug_xwalk.parquet"
DIS_XWALK    = "data/holdout/disease_xwalk.parquet"
WHITELIST    = "data/canon/candidate_whitelist.parquet"
LCC_EXCL     = "data/manifest/a3_lcc_excluded.parquet"
DIS_COLLAPSE = "data/manifest/a6_disease_collapse.parquet"

UPSTREAM_MANIFESTS = [   # sha256 rows inherited into A7's MANIFEST.json
    "data/manifest/okg_manifest.json",
    "data/manifest/a2_canon_manifest.json",
    "data/manifest/broad_manifest.json",
    "data/manifest/repodb_manifest.json",
    "data/manifest/a3_manifest.json",
    "data/manifest/a6_manifest.json",
]

OUT   = Path("data/holdout")
SPLITS = OUT / "splits"
EXCL  = OUT / "inference_exclusions"
for d in (OUT, SPLITS, EXCL):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- frozen constants
US = "\u241f"                     # endpoint_key separator (U+241F). Copy, never rebuild.
DATASET_VERSION = "2.0"
GRAPH_VARIANT   = "full"
SPLIT_SEED      = 20260727        # freeze once; never regenerate "to be safe"
TEST_FRAC       = 0.20
SEED_OFFSET     = {"random": 0, "drug_cold": 1, "disease_cold": 2}
HEADLINE_SPLIT  = "disease_cold"  # the bundle's holdout_positives/negatives = this split

# the 7 leakage (label, relation) pairs, matched on a separator-normalized label
LEAKAGE = [
    ("DRG-DIS", "INDICATION"),        # 57,601
    ("DRG-DIS", "CONTRAINDICATION"),  # 11,718
    ("DRG-DIS", "OFF_LABEL_USE"),     #  1,061
    ("DRG-PHE", "CONTRAINDICATION"),  #  8,279
    ("DRG-PHE", "INDICATION"),        #  1,027
    ("DRG-PHE", "OFF_LABEL_USE"),     #    201
    ("DRG-BPO", "INDICATION"),        #     62
]
EXPECT_REMOVED   = 79_949
EXPECT_DRGDIS    = 70_380            # every row in the label (E3)
EXPECT_EVAL_POS  = 3_969            # A5 evaluable positives
EXPECT_WHITELIST = 1_700

def fatal(msg: str) -> None:
    print(f"[A7 FATAL] {msg}", file=sys.stderr)
    sys.exit(1)

def pick(df: pl.DataFrame, *cands: str) -> str:
    """Return the first column present; FATAL with the real columns if none match."""
    for c in cands:
        if c in df.columns:
            return c
    fatal(f"none of {cands} in columns {df.columns}")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def norm_label(col: pl.Expr) -> pl.Expr:
    return col.str.replace_all("_", "-")

# ============================================================================= STEP 2
# Remove the leakage surface. Whole-label for DRG-DIS (E3); all 7 pairs in full.
# ---------------------------------------------------------------------------------
print("== Step 2: leakage surface -> removed_edge_ids.csv ==")
edges = pl.scan_parquet(CANON_EDGES)

leak = pl.DataFrame(LEAKAGE, schema=["label_n", "relation"], orient="row").lazy()
removed = (
    edges
    .with_columns(label_n=norm_label(pl.col("label")))
    .join(leak, on=["label_n", "relation"], how="inner")
    .select(["from", "to", "label", "relation", "endpoint_key"])
    .collect()
)

if removed.height != EXPECT_REMOVED:
    fatal(f"removed {removed.height:,} edges, expected {EXPECT_REMOVED:,}. "
          f"Read census leakage_surface, do not trust this constant blindly.")

drgdis = removed.filter(norm_label(pl.col("label")) == "DRG-DIS").height
if drgdis != EXPECT_DRGDIS:
    fatal(f"DRG-DIS removed = {drgdis:,}, expected {EXPECT_DRGDIS:,} (E3: the WHOLE label).")

# endpoint_key must be the real U+241F-separated column, not a rebuilt one.
und = removed.filter(norm_label(pl.col("label")) == "DRG-DIS")  # DRG-DIS is undirected
if not und["endpoint_key"].str.contains(US, literal=True).all():
    fatal("endpoint_key on DRG-DIS rows is missing the U+241F separator - "
          "it was rebuilt, not copied from canon_edges. A3 will silently remove 0 edges.")

removed.write_csv(OUT / "removed_edge_ids.csv")
print(f"   removed_edge_ids.csv: {removed.height:,} edges "
      f"({removed.select(['label','relation']).unique().height} types)")

# ============================================================================= disease collapse map
# A6 open rule 3: 21 CUIs -> 2 DIS nodes. Build BOTH a CUI->chosen map (for repoDB keying)
# and a node->chosen map (for graph-derived disease ids), so everything keys consistently.
print("== Loading a6_disease_collapse (21 two-node CUIs) ==")
dc = pl.read_parquet(DIS_COLLAPSE)
cui_col    = pick(dc, "umls_cui")
chosen_col = pick(dc, "chosen_okg_dis_id")
allids_col = pick(dc, "all_okg_dis_ids")

cui_to_chosen = dc.select([cui_col, chosen_col]).rename(
    {cui_col: "umls_cui", chosen_col: "chosen_okg_dis_id"})

node_to_chosen = (
    dc.select([allids_col, chosen_col])
      .explode(allids_col)
      .rename({allids_col: "alt_id", chosen_col: "chosen_okg_dis_id"})
      .unique()
)

def collapse_disease_node(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """Map any alternate disease node to its chosen representative."""
    return (df.join(node_to_chosen, left_on=id_col, right_on="alt_id", how="left")
              .with_columns(pl.coalesce(["chosen_okg_dis_id", id_col]).alias(id_col))
              .drop("chosen_okg_dis_id"))

# ============================================================================= STEP 1
# Evaluable positives / negatives, keyed on (okg_drug_id, okg_disease_id).
#
# ORDER MATTERS. A5's 3,969 was measured on repodb_resolved WITHOUT the collapse
# (STATUS: "the 21 two-node CUIs are not collapsed in repodb_resolved.parquet").
# So we gate on 3,969 PRE-collapse (this proves the join/schema is right), THEN
# collapse the disease column. The collapse MERGES (drug, CUI-via-two-nodes)
# duplicate pairs into one, so the post-collapse count is <= 3,969 BY DESIGN --
# that reduction is exactly the double-count A7 exists to remove, not an error.
# ---------------------------------------------------------------------------------
print("== Step 1: evaluable positives / negatives ==")
res = pl.read_parquet(REPODB_RES)
r_drug = pick(res, "okg_drug_id")
r_dis  = pick(res, "okg_dis_id", "okg_disease_id")
r_db   = pick(res, "drugbank_id")
r_ind  = pick(res, "ind_id", "ind_id_norm")

def evaluable_raw(gold_path: str) -> pl.DataFrame:
    """Evaluable pairs, PRE-collapse (reproduces A5's measurement exactly)."""
    g = pl.read_parquet(gold_path)
    gdb  = pick(g, "drugbank_id")
    gind = pick(g, "ind_id", "ind_id_norm")
    joined = (g.select([gdb, gind]).unique()
               .join(res.select([r_db, r_ind, r_drug, r_dis]),
                     left_on=[gdb, gind], right_on=[r_db, r_ind], how="inner"))
    return (joined
            .filter(pl.col(r_drug).is_not_null() & pl.col(r_dis).is_not_null())
            .select([pl.col(r_drug).alias("okg_drug_id"),
                     pl.col(r_dis).alias("okg_disease_id"),
                     pl.col(gdb).alias("drugbank_id")])
            .unique(["okg_drug_id", "okg_disease_id"]))

# --- gate: reproduce A5 pre-collapse -----------------------------------------
pos_raw = evaluable_raw(GOLD_POS)
neg_raw = evaluable_raw(GOLD_NEG)
if pos_raw.height != EXPECT_EVAL_POS:
    fatal(f"PRE-collapse evaluable positives = {pos_raw.height:,}, expected {EXPECT_EVAL_POS:,} "
          f"(A5). The join/schema is off - check repodb_resolved / gold columns.")
print(f"   pre-collapse positives {pos_raw.height:,} (reproduces A5) | negatives {neg_raw.height:,}")

# --- apply the disease collapse on the disease column, then dedup ------------
def collapsed(df: pl.DataFrame) -> pl.DataFrame:
    return collapse_disease_node(df, "okg_disease_id").unique(["okg_drug_id", "okg_disease_id"])

pos = collapsed(pos_raw)
neg = collapsed(neg_raw)
pos_drop = pos_raw.height - pos.height          # duplicate pairs merged by the collapse

# the reduction MUST be attributable to the 21 collapsed CUIs' alternate nodes,
# never to a schema slip. Every dropped pair had its disease remapped to a chosen node.
alternates = set(node_to_chosen["alt_id"].to_list()) - set(node_to_chosen["chosen_okg_dis_id"].to_list())
touched = pos_raw.filter(pl.col("okg_disease_id").is_in(list(alternates))).height
if pos_drop > touched:
    fatal(f"collapse removed {pos_drop} positive pairs but only {touched} touched a collapsed "
          f"alternate node - the reduction is NOT fully explained by a6_disease_collapse.")
print(f"   post-collapse positives {pos.height:,} (-{pos_drop} duplicate pairs merged) | "
      f"negatives {neg.height:,}")

# a pair resolved as both classes cannot exist (Approved-wins, A5) - assert it
overlap = pos.join(neg, on=["okg_drug_id", "okg_disease_id"], how="inner").height
if overlap:
    fatal(f"{overlap} pairs are in BOTH positives and negatives - Approved-wins violated.")

# annotate positives with inchikey (annotation only; key stays on node ids)
dx = pl.read_parquet(DRUG_XWALK)
dx_drug = pick(dx, "okg_drug_id")
dx_ik   = pick(dx, "inchi_key")
pos = pos.join(dx.select([dx_drug, dx_ik]).rename({dx_drug: "okg_drug_id", dx_ik: "inchikey"}),
               on="okg_drug_id", how="left")
neg = neg.join(dx.select([dx_drug, dx_ik]).rename({dx_drug: "okg_drug_id", dx_ik: "inchikey"}),
               on="okg_drug_id", how="left")
print(f"   evaluable positives {pos.height:,} | negatives {neg.height:,}")

# ============================================================================= STEP 3
# Splits: random (baseline), drug-cold, disease-cold (headline). Temporal dropped.
# ---------------------------------------------------------------------------------
print("== Step 3: splits (random / drug_cold / disease_cold) ==")

def cold_split(entity_col: str, name: str):
    ents = sorted(pos[entity_col].unique().to_list())
    rng = np.random.default_rng(SPLIT_SEED + SEED_OFFSET[name])
    held = set(e for e, keep in zip(ents, rng.random(len(ents)) < TEST_FRAC) if keep)
    hp = pos.filter(pl.col(entity_col).is_in(list(held)))
    hn = neg.filter(pl.col(entity_col).is_in(list(held)))
    return hp, hn, len(held)

def random_split():
    rng = np.random.default_rng(SPLIT_SEED + SEED_OFFSET["random"])
    idx = np.arange(pos.height)
    mask = rng.random(pos.height) < TEST_FRAC
    hp = pos.filter(pl.Series(mask))
    # random-split negatives: a matching fraction of the negative pool
    rng2 = np.random.default_rng(SPLIT_SEED + SEED_OFFSET["random"] + 100)
    hn = neg.filter(pl.Series(rng2.random(neg.height) < TEST_FRAC))
    return hp, hn

splits = {}
hp, hn = random_split();                       splits["random"]       = (hp, hn)
hp, hn, _ = cold_split("okg_drug_id",   "drug_cold");    splits["drug_cold"]    = (hp, hn)
hp, hn, _ = cold_split("okg_disease_id","disease_cold"); splits["disease_cold"] = (hp, hn)

for name, (hp, hn) in splits.items():
    hp.write_csv(SPLITS / f"{name}_holdout_positives.csv")
    hn.write_csv(SPLITS / f"{name}_holdout_negatives.csv")
    print(f"   {name:13s} positives {hp.height:5,} | negatives {hn.height:5,}")

# the frozen bundle uses the headline split
hp, hn = splits[HEADLINE_SPLIT]
hp.write_csv(OUT / "holdout_positives.csv")
hn.write_csv(OUT / "holdout_negatives.csv")
holdout_pairs = set(zip(hp["okg_drug_id"].to_list(), hp["okg_disease_id"].to_list()))

# ============================================================================= STEP 6
# Inference exclusions: known indications MINUS holdout | OBA | LCC-excluded drugs.
# ---------------------------------------------------------------------------------
print("== Step 6: inference exclusions ==")
nodes = pl.read_parquet(CANON_NODES).select([pick(pl.read_parquet(CANON_NODES), "id"), "label"])

# (a) known indications = graph DRG-DIS/INDICATION edges, resolved to (drug, disease) nodes,
#     disease-collapsed, then MINUS the holdout set (D19 - backwards zeroes recall).
ind = removed.filter((norm_label(pl.col("label")) == "DRG-DIS") &
                     (pl.col("relation") == "INDICATION"))
lab_from = nodes.rename({"id": "from", "label": "from_label"})
lab_to   = nodes.rename({"id": "to",   "label": "to_label"})
known = (ind.join(lab_from, on="from").join(lab_to, on="to")
            .with_columns(
                okg_drug_id   =pl.when(pl.col("from_label") == "DRG").then(pl.col("from")).otherwise(pl.col("to")),
                okg_disease_id=pl.when(pl.col("from_label") == "DIS").then(pl.col("from")).otherwise(pl.col("to")))
            .select(["okg_drug_id", "okg_disease_id"]).unique())
known = collapse_disease_node(known, "okg_disease_id").unique()

known_minus = known.filter(
    ~pl.struct(["okg_drug_id", "okg_disease_id"]).map_elements(
        lambda s: (s["okg_drug_id"], s["okg_disease_id"]) in holdout_pairs,
        return_dtype=pl.Boolean))
known_minus.write_csv(EXCL / "known_indications_minus_holdout.csv")

# (b) OBA disease nodes - NOT id STARTS WITH 'OBA' (E19), never id_prefix <> 'OBA'
oba = nodes.filter((pl.col("label") == "DIS") & pl.col("id").str.starts_with("OBA")).select("id")
oba.write_csv(EXCL / "oba_disease_ids.csv")

# (c) LCC-excluded drugs - by ARTIFACT, never by degree (E16)
lcc = pl.read_parquet(LCC_EXCL)
lcc_drugs = lcc.filter(pl.col("label") == "DRG").select(pick(lcc, "id")).rename({"id": "okg_drug_id"}) \
    if "id" in lcc.columns else lcc.filter(pl.col("label") == "DRG").select("id")
lcc_drugs.write_csv(EXCL / "lcc_excluded_drug_ids.csv")
print(f"   known-minus-holdout {known_minus.height:,} | OBA {oba.height:,} | LCC-excluded drugs {lcc_drugs.height:,}")

# ============================================================================= candidate whitelist + crosswalks
print("== candidate_whitelist.csv + crosswalks ==")
wl = pl.read_parquet(WHITELIST)
if wl.height != EXPECT_WHITELIST:
    fatal(f"whitelist = {wl.height:,}, expected {EXPECT_WHITELIST:,} (A4/A6 RATIFIED).")
lcc_ids = set(lcc.filter(pl.col("label") == "DRG")["id"].to_list())
wl_drug_col = pick(wl, "okg_drug_id")
if wl.filter(pl.col(wl_drug_col).is_in(list(lcc_ids))).height:
    fatal("whitelist contains an LCC-excluded drug id - A4/A6 invariant broken.")
wl.write_csv(OUT / "candidate_whitelist.csv")

pl.read_parquet(DRUG_XWALK).write_csv(OUT / "drug_xwalk.csv")
pl.read_parquet(DIS_XWALK).write_csv(OUT / "disease_xwalk.csv")

# ============================================================================= a7_decisions.json
print("== a7_decisions.json ==")
decisions = {
    "stage": "A7",
    "dataset_version": DATASET_VERSION,
    "graph_variant": GRAPH_VARIANT,
    "leakage_surface": {
        "pairs": [{"label": l, "relation": r} for l, r in LEAKAGE],
        "edges_removed": EXPECT_REMOVED,
        "drg_dis_removed": EXPECT_DRGDIS,
        "e3_note": "DRG-DIS is entirely leakage; the whole label is deleted (E3).",
    },
    "contraindication_policy": {
        "DRG-DIS": "whole label removed (INDICATION+CONTRAINDICATION+OFF_LABEL_USE) - E3 governs",
        "DRG-PHE": "removed in FULL (all three relations) - satisfies A3 gate (79,949 / 50 types)",
        "alternative_rejected": ("held-out-only removal of DRG-PHE/CONTRAINDICATION would leave the "
                                 "type alive; it requires relaxing a3_emit_csvs.py:load_removed() and "
                                 "the A3 gate in the same commit. Not taken."),
    },
    "holdout_key": ["okg_drug_id", "okg_disease_id"],
    "holdout_key_rationale": "E15 (repoDB via DrugBank), E18 (69 InChIKey collisions), 984 keyless Broad.",
    "disease_collapse_applied": "a6_disease_collapse.parquet (umls_cui -> chosen_okg_dis_id) before keying",
    "negatives": "Terminated + Withdrawn + Suspended (D21/E7), not Terminated alone",
    "split": {
        "seed": SPLIT_SEED,
        "test_fraction": TEST_FRAC,
        "seed_offsets": SEED_OFFSET,
        "headline": HEADLINE_SPLIT,
        "variants": ["random", "drug_cold", "disease_cold"],
        "temporal": "DROPPED - 2.0 is a property-only refresh; DRG-DIS/INDICATION identical across releases.",
    },
    "inference_exclusions": {
        "known_indications_minus_holdout": "D19 - excluding the holdout too zeroes recall",
        "oba_disease_nodes": "E19 - NOT id STARTS WITH 'OBA', never id_prefix <> 'OBA'",
        "lcc_excluded_drugs": "E16 - by artifact a3_lcc_excluded.parquet, never by degree",
    },
    "evaluable_positives": {
        "a5_precollapse": EXPECT_EVAL_POS,            # 3,969 - reproduced as a gate
        "a7_postcollapse": int(pos.height),           # after a6_disease_collapse dedup
        "duplicate_pairs_merged": int(pos_drop),      # = 3,969 - a7_postcollapse
        "note": ("The 21 two-node CUIs double-counted pairs in repodb_resolved; the collapse "
                 "merges them. a7_postcollapse (< 3,969) is the correct evaluable set, drawn from "
                 "for the splits. The reduction is fully attributable to a6_disease_collapse."),
    },
    "circularity_caveat": {
        "evaluable_positives_a5": EXPECT_EVAL_POS,
        "already_drg_dis_indication_edge": 3752,
        "graph_independent": 217,
        "note": ("94.5% of A5's evaluable positives are the graph's own DrugCentral-sourced "
                 "indication edges. A7 removes them by construction, so this is not model leakage, "
                 "but recovery of the 3,752 is NOT independent validation. Lead the evaluation "
                 "writeup with the ~217. (Measured pre-collapse; the -26 dedup does not change the "
                 "caveat's force.)"),
    },
    "date": date.today().isoformat(),
}
(OUT / "a7_decisions.json").write_text(json.dumps(decisions, indent=2))

# ============================================================================= MANIFEST.json
print("== MANIFEST.json ==")
own_outputs = [
    "removed_edge_ids.csv", "holdout_positives.csv", "holdout_negatives.csv",
    "candidate_whitelist.csv", "drug_xwalk.csv", "disease_xwalk.csv",
    "a7_decisions.json",
]
own = {f: {"sha256": sha256(OUT / f), "bytes": (OUT / f).stat().st_size} for f in own_outputs}
for name, (hp2, hn2) in splits.items():
    for suffix in ("positives", "negatives"):
        p = SPLITS / f"{name}_holdout_{suffix}.csv"
        own[f"splits/{p.name}"] = {"sha256": sha256(p), "bytes": p.stat().st_size}
for p in sorted(EXCL.glob("*.csv")):
    own[f"inference_exclusions/{p.name}"] = {"sha256": sha256(p), "bytes": p.stat().st_size}

inherited = {}
for m in UPSTREAM_MANIFESTS:
    mp = Path(m)
    if mp.exists():
        inherited[mp.name] = {"sha256": sha256(mp), "bytes": mp.stat().st_size}
    else:
        inherited[mp.name] = {"sha256": None, "note": "manifest not found at expected path"}

manifest = {
    "stage": "A7",
    "dataset_version": DATASET_VERSION,   # the DOI does NOT identify the data (A1)
    "graph_variant": GRAPH_VARIANT,
    "split_seed": SPLIT_SEED,
    "headline_split": HEADLINE_SPLIT,
    "date": date.today().isoformat(),
    "outputs": own,
    "inherited_source_manifests": inherited,
}
(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

print("\n== A7 done. Now run: python scripts/a7_verify.py ==")
