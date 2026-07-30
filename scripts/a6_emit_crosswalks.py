#!/usr/bin/env python3
"""
a6_emit_crosswalks.py  —  Emit A6's crosswalk tables + coverage, re-freeze the E23 reach.

Run AFTER a6_collapse_rules.py. Produces:
  data/holdout/drug_xwalk.parquet      okg_drug_id | inchi_key | drugbank_id | broad_pert_iname | name
  data/holdout/disease_xwalk.parquet   okg_dis_id  | umls_cui  | id_prefix   | name | mondo_if_present
  data/manifest/a6_coverage.json       measured coverage with DENOMINATORS (pitfall 7)
  data/manifest/a6_manifest.json       dataset_version + graph_variant + sha256 of outputs

Frozen keys (D10, a6_keys.json) — this script only EMITS them, it does not re-decide:
  Broad  -> OKG drug     : InChIKey        (promoted `inchi_key`)
  repoDB -> OKG drug     : DrugBank        (from properties.source_ids, by SHAPE `DB#####` — E15)
  repoDB -> OKG disease  : UMLS CUI        (promoted `umls_cui`, native only — E23 ruling)

Key the crosswalks on the OKG node id: globally unique, 1:1 by construction (A2 G2).
InChIKey is NOT unique (E18) and Broad `pert_iname` is NOT unique — never key on those.
"""
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path
import polars as pl

CANON_NODES = Path("data/canon/canon_nodes.parquet")
BROAD       = Path("data/canon/broad_compounds.parquet")     # one row per compound (7,540)
REPODB_RES  = Path("data/gold/repodb_resolved.parquet")      # A5: pairs with okg_drug_id/okg_dis_id where resolved
REPODB_PAIR = Path("data/gold/repodb_pairs.parquet")         # A5: 10,602 pairs (for CUI/drugbank denominators)
A6_KEYS     = Path("data/manifest/a6_keys.json")             # frozen reach numbers to cross-check against
A6_DEC      = Path("data/manifest/a6_decisions.json")        # written by a6_collapse_rules.py
A2_MAN      = Path("data/manifest/a2_canon_manifest.json")   # dataset_version + graph_variant

OUT_DRUG_XW = Path("data/holdout/drug_xwalk.parquet")
OUT_DIS_XW  = Path("data/holdout/disease_xwalk.parquet")
OUT_COV     = Path("data/manifest/a6_coverage.json")
OUT_MAN     = Path("data/manifest/a6_manifest.json")

DRUGBANK_SHAPE = r"^DB\d+$"   # E15: DrugBank identified by shape; bare-numeric elements are unattributable


def fatal(msg: str) -> None:
    print(f"[FATAL a6_emit_crosswalks] {msg}", file=sys.stderr)
    sys.exit(1)


def require(p: Path) -> None:
    if not p.exists():
        fatal(f"missing required input: {p}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(p: Path) -> dict:
    require(p)
    return json.loads(p.read_text())


def main() -> None:
    for p in (CANON_NODES, BROAD, A6_DEC):
        require(p)
    nodes = pl.read_parquet(CANON_NODES)

    # =========================================================================
    # DRUG CROSSWALK
    # =========================================================================
    drg = nodes.filter(pl.col("label") == "DRG").select(["id", "inchi_key", "name", "properties_json"])

    # (1) DrugBank from source_ids — decode at a FIXED shape (A2-2 lesson), then filter by DB##### (E15).
    src_schema = pl.Struct({"source_ids": pl.List(pl.String)})
    drg_db = (drg.select(["id", "properties_json"])
                 .with_columns(pl.col("properties_json").str.json_decode(src_schema).alias("_p"))
                 .with_columns(pl.col("_p").struct.field("source_ids").alias("source_ids"))
                 .drop("_p", "properties_json")
                 .explode("source_ids")
                 .filter(pl.col("source_ids").str.contains(DRUGBANK_SHAPE))
                 .rename({"source_ids": "drugbank_id"})
                 .unique(["id", "drugbank_id"]))

    # DrugBank must be 1:1 on the graph side (open rule 5 / E18). Assert, don't assume.
    db_dupe = (drg_db.group_by("drugbank_id").len().filter(pl.col("len") > 1))
    if db_dupe.height:
        print(db_dupe.head(20))
        fatal(f"{db_dupe.height} DrugBank id(s) map to >1 DRG node — STATUS §7 says this is 0. "
              f"Either the graph changed or the shape filter is wrong. Do not emit a broken crosswalk.")

    # (2) Broad pert_iname via InChIKey (primary) + salt fallback on the `inchikeys` list (rule 2).
    broad = pl.read_parquet(BROAD)
    b_primary = (broad.filter(pl.col("inchi_key").is_not_null())
                      .select(["inchi_key", "pert_iname"]).unique("inchi_key"))
    if "inchikeys" in broad.columns:
        b_salt = (broad.filter(pl.col("inchi_key").is_null())
                       .select(["inchikeys", "pert_iname"]).explode("inchikeys")
                       .drop_nulls("inchikeys").rename({"inchikeys": "inchi_key"})
                       .unique("inchi_key"))
        b_key = pl.concat([b_primary, b_salt], how="vertical").unique("inchi_key")
    else:
        b_key = b_primary

    drug_xwalk = (drg.select(pl.col("id").alias("okg_drug_id"), "inchi_key", "name")
                     .join(drg_db.rename({"id": "okg_drug_id"}), on="okg_drug_id", how="left")
                     .join(b_key.rename({"pert_iname": "broad_pert_iname"}),
                           on="inchi_key", how="left")
                     .select(["okg_drug_id", "inchi_key", "drugbank_id", "broad_pert_iname", "name"]))

    # Crosswalk is keyed on okg_drug_id (unique by construction). Assert it.
    if drug_xwalk.height != drug_xwalk["okg_drug_id"].n_unique():
        fatal("drug_xwalk not 1:1 on okg_drug_id — a join fanned out. Check drugbank/pert_iname joins.")

    OUT_DRUG_XW.parent.mkdir(parents=True, exist_ok=True)
    drug_xwalk.write_parquet(OUT_DRUG_XW)
    print(f"[a6] wrote {OUT_DRUG_XW}  ({drug_xwalk.height} DRG nodes; "
          f"{drug_xwalk['drugbank_id'].drop_nulls().n_unique()} distinct DrugBank ids; "
          f"{drug_xwalk['broad_pert_iname'].drop_nulls().len()} Broad-annotated)")

    # =========================================================================
    # DISEASE CROSSWALK
    # =========================================================================
    # Exclude OBA the SAFE way (E19). id_prefix stays as an annotation, never a filter predicate.
    dis = nodes.filter((pl.col("label") == "DIS") & (~pl.col("id").str.starts_with("OBA")))
    disease_xwalk = dis.select(
        okg_dis_id=pl.col("id"),
        umls_cui=pl.col("umls_cui"),
        id_prefix=pl.col("id_prefix"),
        name=pl.col("name"),
        mondo_if_present=pl.when(pl.col("id").str.starts_with("MONDO"))
                           .then(pl.col("id")).otherwise(None),
    )
    if disease_xwalk.height != disease_xwalk["okg_dis_id"].n_unique():
        fatal("disease_xwalk not 1:1 on okg_dis_id.")
    disease_xwalk.write_parquet(OUT_DIS_XW)
    print(f"[a6] wrote {OUT_DIS_XW}  ({disease_xwalk.height} non-OBA DIS nodes)")

    # =========================================================================
    # COVERAGE — always with denominators (pitfall 7). Cross-check against frozen numbers.
    # =========================================================================
    cov: dict = {"note": "denominators explicit; these are measured boundaries, not failures"}

    # -- disease reach (E23): native `umls_cui` ONLY, over repoDB's distinct CUIs --
    if REPODB_PAIR.exists():
        repodb_cuis = pl.read_parquet(REPODB_PAIR).select("ind_id").unique().drop_nulls()
        n_cuis = repodb_cuis.height
        dis_keys = disease_xwalk.select("umls_cui").drop_nulls().unique()
        reach = repodb_cuis.join(dis_keys, left_on="ind_id", right_on="umls_cui", how="inner").height
        cov["repodb_disease_reach_native_umls_cui"] = {
            "reached": reach, "of": n_cuis,
            "pct": round(100 * reach / n_cuis, 1) if n_cuis else None,
            "ruling": "native = promoted umls_cui only; concept_ids NOT counted (E23)",
        }
        # E23 reconciliation: expect 717 (A5 native), NOT the frozen 719. Tolerance ±5 (A5-3 precedent).
        if abs(reach - 717) > 5:
            print(f"[a6][WARN] disease reach {reach} deviates from A5 native 717 by >5 — investigate before freezing.")
        # write the re-frozen reach back into the decisions file
        dec = load_json(A6_DEC)
        dec.setdefault("e23_native_key_ruling", {})["frozen_disease_reach"] = {"reached": reach, "of": n_cuis}
        A6_DEC.write_text(json.dumps(dec, indent=2))
        print(f"[a6] E23 re-froze disease reach at {reach}/{n_cuis} in {A6_DEC}")
    else:
        print(f"[a6][WARN] {REPODB_PAIR} absent — disease reach not re-measured (E23 left unresolved).")

    # -- drug reach (repoDB): DrugBank via source_ids; cross-check A5's 1,905 --
    if REPODB_RES.exists():
        res = pl.read_parquet(REPODB_RES)
        if "drugbank_id" in res.columns:
            db_denom = res.select("drugbank_id").unique().drop_nulls().height
            db_reach = res.filter(pl.col("okg_drug_id").is_not_null()).select("drugbank_id").unique().height
            cov["repodb_drug_reach_drugbank"] = {"reached": db_reach, "of": db_denom,
                                                 "pct": round(100 * db_reach / db_denom, 1) if db_denom else None}
            if abs(db_reach - 1905) > 5:
                print(f"[a6][WARN] repoDB drug reach {db_reach} deviates from frozen 1,905 by >5.")

    # -- Broad launched reach (InChIKey) — cross-check A4's 1,700 whitelist / 63.8% --
    launched = broad.filter(pl.col("clinical_phase") == "Launched")
    drg_keys = drg.select("inchi_key").drop_nulls().unique()
    launched_reach = (launched.filter(pl.col("inchi_key").is_not_null())
                              .join(drg_keys, on="inchi_key", how="inner")
                              .select("inchi_key").unique().height)
    cov["broad_launched_reach_inchikey"] = {"reached_keys": launched_reach, "of_launched": launched.height,
                                            "note": "pre-LCC key reach; final whitelist is 1,700 nodes (A4, E22)"}

    # -- disease space shape (report the honest denominator, not '36,345 diseases') --
    dis_all = nodes.filter(pl.col("label") == "DIS")
    oba = dis_all.filter(pl.col("id").str.starts_with("OBA")).height
    cov["disease_space"] = {
        "dis_nodes_total": dis_all.height,
        "oba_excluded": oba,
        "non_oba": disease_xwalk.height,
        "with_umls_cui": disease_xwalk.select("umls_cui").drop_nulls().height,
        "caveat": "only ~6.8% of DIS nodes touch a drug (STATUS §4); '36,345 diseases' overstates ~14x",
    }

    OUT_COV.write_text(json.dumps(cov, indent=2))
    print(f"[a6] wrote {OUT_COV}")

    # =========================================================================
    # MANIFEST — dataset_version + graph_variant + sha256 of the two crosswalks
    # =========================================================================
    a2 = load_json(A2_MAN) if A2_MAN.exists() else {}
    manifest = {
        "stage": "A6",
        "dataset_version": a2.get("dataset_version", "UNKNOWN"),
        "graph_variant": a2.get("graph_variant", "UNKNOWN"),
        "outputs": {
            str(OUT_DRUG_XW): {"rows": drug_xwalk.height, "sha256": sha256(OUT_DRUG_XW)},
            str(OUT_DIS_XW):  {"rows": disease_xwalk.height, "sha256": sha256(OUT_DIS_XW)},
        },
        "keys": "frozen in a6_keys.json (D10); this stage emits, does not re-decide",
        "collapse_rules": str(A6_DEC),
    }
    if manifest["dataset_version"] == "UNKNOWN":
        print("[a6][WARN] dataset_version UNKNOWN — a2_canon_manifest.json not found; fill before A7 MANIFEST.")
    OUT_MAN.write_text(json.dumps(manifest, indent=2))
    print(f"[a6] wrote {OUT_MAN}")
    print("[a6] crosswalks emitted. Next: a6_verify.py, then A7.")


if __name__ == "__main__":
    main()
