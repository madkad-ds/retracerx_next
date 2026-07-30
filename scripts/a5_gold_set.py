#!/usr/bin/env python3
"""
A5 — repoDB gold set (repoDB alone; drugrepurposing.info dropped, D11).

Builds the evaluation gold set from repoDB, resolves it to OptimusKG for a COVERAGE
measurement (the frozen crosswalk emission itself is A6's job, not A5's), and emits the
positives/negatives plus decision + manifest files that A6/A7 inherit.

Consumes
  <repodb>                              repoDB export, pinned at A2  (default data/sources/repodb_full.csv)
  data/canon/canon_nodes.parquet        A2 canonical nodes (for OKG resolution — coverage only)
  data/manifest/repodb_manifest.json    expected row/pair/status counts  (GATE source, never prose)
  data/manifest/a6_keys.json            frozen reach 1,905 drugs / 719 CUIs  (GATE source)

Emits
  data/gold/repodb_pairs.parquet        10,602 pairs, status resolved (Approved wins)
  data/gold/gold_positives.parquet
  data/gold/gold_negatives.parquet
  data/gold/repodb_resolved.parquet     pairs annotated with okg_drug_id / okg_dis_id (for A5 circularity + A7)
  data/manifest/a5_decisions.json       status-conflict rule + counts (the warrant)
  data/manifest/a5_manifest.json        sha256 + counts

Design rules obeyed here (from STATUS.md):
  * E8  rows are per-TRIAL — collapse to unique (drugbank_id, ind_id) pairs.
  * E7  status has FOUR levels — negative class = Terminated + Withdrawn + Suspended.
  * E15 repoDB DrugBank IDs live in OKG properties.source_ids (unlabeled list; DB##### by shape).
  * E9  repoDB ships NO DrugCentral struct_id — do not attempt that shortcut.
  * open-rule-3  21 CUIs map to 2 DIS nodes — do NOT collapse here (A6 owns it); flag and defer.
  * Gates read expected numbers from manifests, never from a copied sentence (failure log A3-1/A4-4).
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
import polars as pl

# ----- status vocabulary (E7) -------------------------------------------------
POSITIVE = ["Approved"]
NEGATIVE = ["Terminated", "Withdrawn", "Suspended"]   # NOT Terminated alone
CONFLICT_RULE = "Approved wins"

DB_SHAPE = re.compile(r"^DB\d{5}$")   # DrugBank accession by SHAPE (E15); bare numerics are unattributable


# ----- small helpers ----------------------------------------------------------
def fatal(msg: str) -> None:
    print(f"[A5 FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dig(obj, path: list[str], label: str):
    """Fetch a nested value; FATAL loudly listing the real structure on a miss (cf. A3-2).
    Adjust `path` ONCE for your manifest layout — the FATAL prints what is actually there."""
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            here = list(cur.keys()) if isinstance(cur, dict) else type(cur).__name__
            fatal(f"{label}: path {path!r} missing at {k!r}. Available here: {here}")
        cur = cur[k]
    return cur


def drugbank_from_props(pj: str) -> list[str]:
    """Extract DrugBank accessions from a node's properties_json.source_ids (E15).
    source_ids is an UNLABELED list e.g. ['DB00341','3561']; keep only DB#####-shaped elements."""
    try:
        props = json.loads(pj)
    except Exception:
        return []
    v = props.get("source_ids")
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    return [x for x in v if isinstance(x, str) and DB_SHAPE.match(x)]


# ----- main -------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repodb", default="data/sources/repodb_full.csv")
    ap.add_argument("--nodes",  default="data/canon/canon_nodes.parquet")
    ap.add_argument("--repodb-manifest", default="data/manifest/repodb_manifest.json")
    ap.add_argument("--a6-keys",         default="data/manifest/a6_keys.json")
    ap.add_argument("--out",    default="data/gold")
    ap.add_argument("--manifest-dir", default="data/manifest")
    args = ap.parse_args()

    repodb_path = Path(args.repodb)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = Path(args.manifest_dir); man.mkdir(parents=True, exist_ok=True)

    # 0. Pin check — repoDB was pinned at A2. Fail loudly on drift. -------------
    rman = json.loads(Path(args.repodb_manifest).read_text())
    exp_sha  = dig(rman, ["sha256"], "repodb_manifest")           # adjust path if your layout differs
    got_sha  = sha256_file(repodb_path)
    if got_sha != exp_sha:
        fatal(f"repoDB sha256 drift: manifest {exp_sha[:12]}… != file {got_sha[:12]}…")
    exp_rows  = int(dig(rman, ["rows"], "repodb_manifest"))
    exp_pairs = int(dig(rman, ["unique_drug_indication_pairs"], "repodb_manifest"))

    # 1. Load; columns are exactly drug_name,drugbank_id,ind_name,ind_id,NCT,status,phase,DetailedStatus
    repodb = pl.read_csv(repodb_path)
    if repodb.height != exp_rows:
        fatal(f"row count {repodb.height:,} != manifest {exp_rows:,} (per-trial rows, E8)")
    for col in ("drugbank_id", "ind_id", "status", "phase"):
        if col not in repodb.columns:
            fatal(f"missing column {col!r}; have {repodb.columns}")

    # 2. Collapse per-TRIAL -> per-PAIR (E8). ind_id IS the UMLS CUI. -----------
    pairs = (repodb
             .group_by(["drugbank_id", "ind_id"])
             .agg(
                 pl.col("status").unique().sort().alias("statuses"),
                 pl.len().alias("n_trials"),
                 pl.col("phase").unique().sort().alias("phases"),
                 pl.col("drug_name").first().alias("drug_name"),
                 pl.col("ind_name").first().alias("ind_name"),
             ))
    if pairs.height != exp_pairs:
        fatal(f"pair count {pairs.height:,} != manifest {exp_pairs:,} "
              f"(do NOT compare to the 2017 paper's 6,677/4,123 — E8)")

    # Normalize the disease join key (strip whitespace) before matching — how-to A6 rule.
    pairs = pairs.with_columns(pl.col("ind_id").str.strip_chars().alias("ind_id_norm"))

    # 3. Resolve status conflicts (Approved wins) and classify (E7). ------------
    has_pos = pl.col("statuses").list.eval(pl.element().is_in(POSITIVE)).list.any()
    has_neg = pl.col("statuses").list.eval(pl.element().is_in(NEGATIVE)).list.any()
    pairs = pairs.with_columns(
        pl.when(has_pos).then(pl.lit("positive"))
          .when(has_neg).then(pl.lit("negative"))
          .otherwise(pl.lit("UNRESOLVED")).alias("resolved_class"),
        (pl.col("statuses").list.len() > 1).alias("status_conflict"),
    )
    stray = pairs.filter(pl.col("resolved_class") == "UNRESOLVED")
    if stray.height:
        fatal(f"{stray.height} pairs carry a status outside {POSITIVE+NEGATIVE}; "
              f"E7 says there are exactly four levels — investigate before proceeding")

    n_conflict = int(pairs.select(pl.col("status_conflict").sum()).item())
    positives  = pairs.filter(pl.col("resolved_class") == "positive")
    negatives  = pairs.filter(pl.col("resolved_class") == "negative")
    print(f"pairs={pairs.height:,}  positives={positives.height:,}  "
          f"negatives={negatives.height:,}  conflicts(Approved-wins)={n_conflict:,}")

    # 4. Resolve to OKG for COVERAGE (frozen keys; A6 emits the crosswalk itself)
    nodes = pl.read_parquet(args.nodes)

    #    drug: DrugBank via properties.source_ids (E15). DrugBank-side duplication is 0 (E18) — assert.
    drg = nodes.filter(pl.col("label") == "DRG").select(["id", "properties_json"])
    drug_map = (drg.with_columns(
                    pl.col("properties_json")
                      .map_elements(drugbank_from_props, return_dtype=pl.List(pl.Utf8))
                      .alias("db"))
                   .explode("db").drop_nulls("db")
                   .select(pl.col("id").alias("okg_drug_id"), pl.col("db").alias("drugbank_id"))
                   .unique())
    dup_db = drug_map.group_by("drugbank_id").len().filter(pl.col("len") > 1)
    if dup_db.height:
        fatal(f"{dup_db.height} DrugBank IDs map to >1 OKG node — contradicts E18 "
              f"(DrugBank duplication = 0); a drug-side collapse rule would be required")

    #    disease: umls_cui column. 21 CUIs -> 2 DIS nodes (open rule 3) — flag, do NOT collapse.
    dis = (nodes.filter(pl.col("label") == "DIS")
                .select(pl.col("id").alias("okg_dis_id"),
                        pl.col("umls_cui").str.strip_chars().alias("umls_cui"))
                .drop_nulls("umls_cui"))
    cui_multi = dis.group_by("umls_cui").len().filter(pl.col("len") > 1)
    print(f"[open-rule-3] CUIs mapping to >1 DIS node: {cui_multi.height} "
          f"(expected ~21 EFO+MONDO pairs) — deferred to A6, not collapsed here")

    #    Coverage against the FROZEN reach in a6_keys.json (gate, not prose). ---
    keys = json.loads(Path(args.a6_keys).read_text())
    exp_drug_reach = int(dig(keys, ["keys", "repodb_drug",    "coverage", "reachable_via_properties"],   "a6_keys"))
    exp_dis_reach  = int(dig(keys, ["keys", "repodb_disease", "coverage", "reachable_native_and_xrefs"], "a6_keys"))
    exp_drug_total = int(dig(keys, ["keys", "repodb_drug",    "coverage", "repodb_drugs"], "a6_keys"))
    exp_dis_total  = int(dig(keys, ["keys", "repodb_disease", "coverage", "repodb_cuis"],  "a6_keys"))

    drug_total = pairs["drugbank_id"].n_unique()
    dis_total  = pairs["ind_id"].n_unique()
    if drug_total != exp_drug_total:
        fatal(f"distinct repoDB drugs {drug_total} != a6_keys {exp_drug_total} — the collapse changed the population")
    if dis_total != exp_dis_total:
        fatal(f"distinct repoDB CUIs {dis_total} != a6_keys {exp_dis_total} — the collapse changed the population")

    drug_reach = (pairs.select("drugbank_id").unique()
                       .join(drug_map.select("drugbank_id").unique(), on="drugbank_id", how="semi").height)
    dis_reach  = (pairs.select("ind_id_norm").unique()
                       .join(dis.select(pl.col("umls_cui")).unique(),
                             left_on="ind_id_norm", right_on="umls_cui", how="semi").height)
    if drug_reach != exp_drug_reach:
        fatal(f"drug reach {drug_reach} != a6_keys {exp_drug_reach} — source_ids extraction changed")
    dis_delta = dis_reach - exp_dis_reach
    if abs(dis_delta) > 5:
        fatal(f"disease reach {dis_reach} != a6_keys {exp_dis_reach} (delta {dis_delta:+d}) — umls_cui join changed")
    if dis_delta:
        print(f"[WARN] disease reach {dis_reach} vs a6_keys {exp_dis_reach} ({dis_delta:+d}) — within "
              f"tolerance; A5 records its measured value. Normalization nuance vs a5_key_probe, not drift.")
    print(f"reach: drug {drug_reach}/{drug_total} (80.0%)  "
          f"disease {dis_reach}/{dis_total} (~49% — honest ceiling, bounded by the graph)")

    #    Annotate every pair with OKG ids where they resolve (LEFT joins keep unresolved rows).
    #    NB: the disease left-join can duplicate a pair across the 21 two-node CUIs — that is
    #    intentional here (nothing is collapsed); A6/A7 apply the collapse rule before freezing.
    resolved = (pairs
                .join(drug_map, on="drugbank_id", how="left")
                .join(dis, left_on="ind_id_norm", right_on="umls_cui", how="left"))

    # evaluable positives = positive pairs where BOTH sides resolve (A7's definition; measured here)
    pos_res = (positives
               .join(drug_map, on="drugbank_id", how="inner")
               .join(dis, left_on="ind_id_norm", right_on="umls_cui", how="inner"))
    evaluable_pos = pos_res.select(["okg_drug_id", "okg_dis_id"]).unique().height
    print(f"evaluable positives (distinct OKG pairs, both sides resolve): {evaluable_pos:,} "
          f"— will look artificially high; see circularity (A5 step 5)")

    # 5. Emit ------------------------------------------------------------------
    pairs.write_parquet(out / "repodb_pairs.parquet")
    positives.write_parquet(out / "gold_positives.parquet")
    negatives.write_parquet(out / "gold_negatives.parquet")
    resolved.write_parquet(out / "repodb_resolved.parquet")

    decisions = {
        "stage": "A5",
        "source": "repoDB only (drugrepurposing.info dropped — D11)",
        "status_conflict_rule": CONFLICT_RULE,
        "status_conflict_count": n_conflict,
        "positive_class": POSITIVE,
        "negative_class": NEGATIVE,
        "n_pairs": pairs.height,
        "n_positives": positives.height,
        "n_negatives": negatives.height,
        "resolution_keys": {
            "drug": "DrugBank via properties.source_ids (shape DB#####) — E15",
            "disease": "umls_cui — E9/D10",
        },
        "coverage": {
            "drug_reach": drug_reach, "drug_total": pairs["drugbank_id"].n_unique(),
            "disease_reach": dis_reach, "disease_total": pairs["ind_id"].n_unique(),
            "evaluable_positive_okg_pairs": evaluable_pos,
        },
        "deferred_to_A6": {
            "open_rule_3_cui_multi_node_groups": cui_multi.height,
            "note": "disease-side collapse (EFO+MONDO) is A6's; nothing collapsed in A5",
        },
    }
    (man / "a5_decisions.json").write_text(json.dumps(decisions, indent=2))

    outputs = ["repodb_pairs.parquet", "gold_positives.parquet",
               "gold_negatives.parquet", "repodb_resolved.parquet"]
    manifest = {
        "stage": "A5",
        "inputs": {
            "repodb": {"path": str(repodb_path), "sha256": got_sha},
            "nodes":  {"path": args.nodes, "sha256": sha256_file(Path(args.nodes))},
        },
        "outputs": {name: {"sha256": sha256_file(out / name),
                           "rows": pl.read_parquet(out / name).height} for name in outputs},
        "counts": {"pairs": pairs.height, "positives": positives.height,
                   "negatives": negatives.height, "conflicts": n_conflict,
                   "evaluable_positive_okg_pairs": evaluable_pos},
    }
    (man / "a5_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[A5] gold set emitted -> data/gold/ ; decisions + manifest written. "
          "Run a5_circularity.py next.")


if __name__ == "__main__":
    main()
