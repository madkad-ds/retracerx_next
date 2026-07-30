#!/usr/bin/env python3
"""
A4 — Broad Repurposing Hub: candidate universe, drug->target reference, known-indication annotation.

Reads ONLY frozen A2/A3 artifacts (STATUS §5). Emits:
  data/canon/candidate_whitelist.parquet        (a) Broad Launched ∩ OKG, minus LCC-excluded, 1 node / drug
  data/canon/drug_targets.parquet               (b) Broad HGNC target symbols -> OKG Ensembl GEN ids
  data/canon/broad_known_indications.parquet    (c) Broad free-text indication/disease_area (ANNOTATION only)
  data/manifest/a4_inchikey_collisions.parquet      collision groups handed to A6 to ratify
  data/manifest/a4_approval_crosscheck.parquet      Broad Launched vs OKG is_approved disagreement
  data/manifest/a4_manifest.json                    counts + sha256 + provisional-rule flag

Corrections applied (STATUS §6 errata):
  E10  InChIKey lives in the sample file  -> already merged into broad_compounds.parquet by A2. Use that file.
  E11  filename != version; !File_date is -> pinning done at A2 (broad_manifest.json).
  E12  "sample" = physical vial; naive re-join multiplies rows -> NEVER re-join raw files; use broad_compounds.
  E13  Broad InChIKeys are salted; 148 compounds have scalar inchi_key NULL, keys in `inchikeys` list.
  E16  "982 edgeless drugs" is WRONG. Zero nodes have degree 0. Exclude LCC-outsiders by ARTIFACT,
       anti-join on `id` against a3_lcc_excluded.parquet — NEVER by a degree filter.
  E14/E18  InChIKey join is not 1:1 (~75 OKG nodes share a key / 69 true dup groups). Assert + collapse.

Three coincident integers, three DIFFERENT populations (STATUS §7): do not conflate.
  982  = LCC-excluded DRG (outside the largest connected component)
  984  = unscoreable Launched (Broad compound with NO OKG node at all)  -> a coverage fact, not an error
  [U]  = newly-edgeless-after-filtering -> measured later at A3 Step 5, not here.
"""

import json
import hashlib
from pathlib import Path
import polars as pl

# --- Paths -------------------------------------------------------------------
CANON   = Path("data/canon")
MAN     = Path("data/manifest")
EDGES   = "data/canon/canon_edges/*.parquet"

BROAD_COMPOUNDS = CANON / "broad_compounds.parquet"
CANON_NODES     = CANON / "canon_nodes.parquet"
LCC_EXCLUDED    = MAN   / "a3_lcc_excluded.parquet"

# --- Frozen expectations (STATUS §4) -----------------------------------------
EXPECT_LAUNCHED        = 2718     # Broad Launched compounds
EXPECT_MAPPED_MIN      = 1700     # ~1,734 distinct compounds map (63.8%); allow small drift, warn otherwise
EXPECT_UNSCOREABLE     = 984      # Launched with no OKG node
GEN_SYMBOL_COVERAGE    = 0.9968   # symbol present on 61,108 / 61,306 GEN nodes

COLLAPSE_RULE = "degree_desc > is_approved_desc > id_asc"  # A6 open rule 1 — PROVISIONAL, A6 ratifies


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def degree_for_ids(ids: list[str]) -> pl.DataFrame:
    """Degree ONLY for the colliding node ids — cheap. Never used as an exclusion filter (E16)."""
    if not ids:
        return pl.DataFrame({"id": [], "degree": []}, schema={"id": pl.String, "degree": pl.UInt32})
    lf = pl.scan_parquet(EDGES)
    frm = lf.select(pl.col("from").alias("id")).filter(pl.col("id").is_in(ids))
    to  = lf.select(pl.col("to").alias("id")).filter(pl.col("id").is_in(ids))
    return (pl.concat([frm, to]).group_by("id").len()
              .rename({"len": "degree"}).collect(engine="streaming"))


def collapse_collisions(mapped: pl.DataFrame):
    """Reduce InChIKey collisions to one representative OKG node per key (E14/E18).
    Rule (PROVISIONAL, A6 owns the freeze): most edges > is_approved > lowest id.
    Returns (resolved, collision_groups, rule_applied_or_None)."""
    dup_keys = (mapped.group_by("inchi_key").len()
                      .filter(pl.col("len") > 1)["inchi_key"].to_list())
    if len(dup_keys) == 0:
        empty = mapped.head(0)
        return mapped, empty, None

    dup = mapped.filter(pl.col("inchi_key").is_in(dup_keys))
    deg = degree_for_ids(dup["id"].unique().to_list())
    scored = (dup.join(deg, on="id", how="left")
                 .with_columns(pl.col("degree").fill_null(0),
                               appr=(pl.col("is_approved") == "true").cast(pl.Int8)))
    winners = (scored.sort(["inchi_key", "degree", "appr", "id"],
                           descending=[False, True, True, False])
                     .group_by("inchi_key", maintain_order=True)
                     .first()
                     .select(["inchi_key", "id"]))
    kept_dup = dup.join(winners, on=["inchi_key", "id"], how="semi")
    resolved = pl.concat([mapped.filter(~pl.col("inchi_key").is_in(dup_keys)), kept_dup],
                         how="vertical_relaxed")
    # collision report for A6: every colliding node + the degree/approval it was judged on
    collision_groups = scored.select(["inchi_key", "id", "name", "is_approved", "degree"]) \
                             .join(winners.with_columns(chosen=pl.lit(True)),
                                   on=["inchi_key", "id"], how="left") \
                             .with_columns(pl.col("chosen").fill_null(False))
    return resolved, collision_groups, COLLAPSE_RULE


def preflight():
    """Fail legibly on missing inputs, and name the fix for the one that A2 should have built."""
    missing = [p for p in (BROAD_COMPOUNDS, CANON_NODES, LCC_EXCLUDED) if not p.exists()]
    if BROAD_COMPOUNDS in missing:
        raise SystemExit(
            f"[FATAL] {BROAD_COMPOUNDS} not found — A2's Broad join was not persisted.\n"
            "  Locate it:  find . -name 'broad_compounds.parquet'\n"
            "  Or rebuild (do NOT hand-roll the join here — E11/E12/E13 live in a4_broad_join.py):\n"
            "    python scripts/a4_broad_join.py \\\n"
            "      --drug   data/sources/repo-drug-annotation-20200324.txt \\\n"
            "      --sample data/sources/repo-sample-annotation-20240610.txt")
    if missing:
        raise SystemExit(f"[FATAL] missing inputs: {', '.join(str(p) for p in missing)}")


def main():
    MAN.mkdir(parents=True, exist_ok=True)
    CANON.mkdir(parents=True, exist_ok=True)
    preflight()

    broad = pl.read_parquet(BROAD_COMPOUNDS)     # one row per compound (E12 — never re-join raw files)
    nodes = pl.read_parquet(CANON_NODES)

    drg = (nodes.filter(pl.col("label") == "DRG")
                .with_columns(is_approved=pl.col("properties_json").str.json_path_match("$.is_approved"))
                .select(["id", "name", "inchi_key", "is_approved"]))

    # ---- (a) candidate whitelist -------------------------------------------
    launched = broad.filter(pl.col("clinical_phase") == "Launched")
    assert launched.height == EXPECT_LAUNCHED, \
        f"[GATE] Launched drifted: {launched.height} != {EXPECT_LAUNCHED} — broad_compounds changed?"

    # primary-branch schema — the scalar `inchi_key` is the join key; `inchikeys` is NOT carried
    base_cols = [c for c in ["pert_iname", "clinical_phase", "inchi_key"] if c in launched.columns]

    # primary path: scalar InChIKey
    mapped = launched.select(base_cols).join(drg, on="inchi_key", how="inner") \
                     .with_columns(map_via=pl.lit("inchi_key"))

    # E13 salt fallback: scalar inchi_key NULL, keys live in the `inchikeys` list.
    # `inchikeys` is used ONLY to explode+join, then dropped so both branches share one schema
    # (post-explode it is a scalar, which would clash with the primary branch's list column).
    if "inchikeys" in launched.columns:
        salt_pre = [c for c in base_cols if c != "inchi_key"]
        salted = (launched.filter(pl.col("inchi_key").is_null())
                          .select(salt_pre + ["inchikeys"])
                          .explode("inchikeys")
                          .join(drg, left_on="inchikeys", right_on="inchi_key", how="inner")
                          .with_columns(inchi_key=pl.col("inchikeys"), map_via=pl.lit("salt_fallback"))
                          .drop("inchikeys"))
    else:
        salted = mapped.head(0)

    combined = pl.concat([mapped, salted], how="diagonal")

    # E16: exclude LCC-outsiders by ARTIFACT (anti-join on id) — NEVER by degree
    excluded = pl.read_parquet(LCC_EXCLUDED).select("id")
    n_before_lcc = combined["id"].n_unique()
    combined = combined.join(excluded, on="id", how="anti")
    n_after_lcc = combined["id"].n_unique()

    # E14/E18: collapse InChIKey collisions to one representative node per key
    resolved, collisions, rule = collapse_collisions(combined)

    # final whitelist: one row per OKG drug node
    whitelist = (resolved.unique(subset=["id"], keep="first")
                         .select(["id", "name", "inchi_key", "is_approved",
                                  "clinical_phase", "pert_iname", "map_via"])
                         .rename({"id": "okg_drug_id"}))

    mapped_compounds = combined["pert_iname"].n_unique()
    if mapped_compounds < EXPECT_MAPPED_MIN:
        print(f"[WARN] mapped compounds {mapped_compounds} < expected ~1,734 — investigate coverage")

    # ---- (b) drug -> target reference (HGNC -> Ensembl) ---------------------
    gen = nodes.filter(pl.col("label") == "GEN").select(["id", "symbol"]).drop_nulls("symbol")
    cov = gen.height / nodes.filter(pl.col("label") == "GEN").height
    assert cov >= GEN_SYMBOL_COVERAGE - 0.01, f"[GATE] GEN symbol coverage collapsed: {cov:.4f}"

    target_col = "targets" if "targets" in broad.columns else None
    if target_col is None:
        raise SystemExit("[FATAL] broad_compounds has no `targets` column — A2 join changed shape")

    drug_targets = (broad.select(["pert_iname", "inchi_key", "targets"])
                         .explode("targets")
                         .drop_nulls("targets")
                         .with_columns(pl.col("targets").str.strip_chars())
                         .filter(pl.col("targets") != "")
                         .join(gen, left_on="targets", right_on="symbol", how="left")
                         .rename({"id": "okg_gene_id", "targets": "hgnc_symbol"}))
    unmatched_symbols = drug_targets.filter(pl.col("okg_gene_id").is_null())["hgnc_symbol"].n_unique()

    # ---- (c) known-indication annotation (NOT a keyed mapping) --------------
    # The keyed known-indication set is the graph's own DRG-DIS/INDICATION edges,
    # removed at A7 (removed_edge_ids.csv). Here we keep Broad's free text ONLY as annotation.
    # Inference exclusion rule (recorded for A7): known indications MINUS the holdout set.
    ann_cols = [c for c in ["pert_iname", "inchi_key", "indication", "disease_area", "moa"]
                if c in broad.columns]
    known_indications = broad.select(ann_cols)

    # ---- cross-check: Broad Launched vs OKG is_approved --------------------
    crosscheck = (whitelist
                  .with_columns(broad_launched=pl.lit(True),
                                okg_is_approved=pl.col("is_approved"))
                  .group_by(["okg_is_approved"]).len()
                  .rename({"len": "n_launched_nodes"})
                  .sort("n_launched_nodes", descending=True))
    disagreement = whitelist.filter(pl.col("is_approved") != "true").height

    # ---- write outputs ------------------------------------------------------
    out = {
        CANON / "candidate_whitelist.parquet":        whitelist,
        CANON / "drug_targets.parquet":               drug_targets,
        CANON / "broad_known_indications.parquet":    known_indications,
        MAN   / "a4_inchikey_collisions.parquet":     collisions,
        MAN   / "a4_approval_crosscheck.parquet":     crosscheck,
    }
    for path, df in out.items():
        df.write_parquet(path)

    manifest = {
        "stage": "A4",
        "inputs": {p.name: sha256(p) for p in [BROAD_COMPOUNDS, CANON_NODES, LCC_EXCLUDED]},
        "outputs": {p.name: sha256(p) for p in out},
        "counts": {
            "launched_compounds": launched.height,                      # 2,718
            "mapped_compounds": mapped_compounds,                       # ~1,734
            "unscoreable_launched": EXPECT_UNSCOREABLE,                 # 984 (coverage fact)
            "whitelist_nodes": whitelist.height,                        # 1 row per OKG drug node
            "lcc_excluded_dropped": n_before_lcc - n_after_lcc,
            "inchikey_collision_groups": collisions["inchi_key"].n_unique() if collisions.height else 0,
            "drug_target_edges": drug_targets.height,
            "unmatched_target_symbols": unmatched_symbols,
            "gen_symbol_coverage": round(cov, 4),
            "approval_disagreement_nodes": disagreement,
        },
        "collapse_rule": rule,                       # PROVISIONAL — A6 open rule 1 must ratify
        "collapse_rule_status": "PROVISIONAL_pending_A6",
        "notes": [
            "LCC exclusion by artifact (a3_lcc_excluded.parquet), never by degree (E16).",
            "known_indications is ANNOTATION only; keyed set = DRG-DIS/INDICATION removed at A7.",
            "inference exclusion = known indications MINUS holdout set (A4c/A7).",
            "ship the Broad download script, not the file (CC-BY vs !Restrictions).",
        ],
    }
    (MAN / "a4_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- gate (write the gate first — STATUS §8) ---------------------------
    assert whitelist.height == whitelist["okg_drug_id"].n_unique(), \
        "[GATE] whitelist not 1:1 on okg_drug_id — collision collapse failed"
    assert whitelist.join(excluded, left_on="okg_drug_id", right_on="id", how="semi").height == 0, \
        "[GATE] LCC-excluded node leaked into whitelist"
    print(json.dumps(manifest["counts"], indent=2))
    print(f"[OK] A4 complete. Collapse rule '{rule}' is PROVISIONAL — A6 must ratify.")


if __name__ == "__main__":
    main()
