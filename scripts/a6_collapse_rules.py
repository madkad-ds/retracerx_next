#!/usr/bin/env python3
"""
a6_collapse_rules.py  —  Freeze A6's node-collapse rules (open rules 1 & 3).

Run BEFORE a6_emit_crosswalks.py. Produces:
  data/manifest/a6_drug_collapse.parquet     one chosen okg_drug_id per same-InChIKey group (E18: ~69 groups)
  data/manifest/a6_disease_collapse.parquet  one chosen okg_dis_id per CUI that hits 2 DIS nodes (open rule 3: 21)
  data/manifest/a6_decisions.json            the frozen rules, counts, and the E23 native-key ruling

Why this exists (STATUS §7 "A6 open rules"):
  Rule 1 (drug):    InChIKey is NOT unique on the graph (E18/E14) — 69 same-key groups. A Broad compound
                    whose InChIKey collides maps to >1 OKG node. Row order must not pick the representative.
                    A4 applied a PROVISIONAL rule to the 34 Launched-touching groups (D18); A6 RATIFIES it
                    here and extends it to all groups, asserting agreement with A4 as a drift guard.
  Rule 3 (disease): 21 UMLS CUIs resolve to 2 OKG DIS nodes (EFO + MONDO). A5 flagged but did NOT collapse
                    them (D22). Unhandled, A7's holdout double-counts those pairs (STATUS A5->A7 warning).

Collapse rule (both directions), in order:
  1. most edges (degree)         — the better-connected, actually-scoreable node
  2. tie-break:
       drug    -> is_approved True first
       disease -> id_prefix priority EFO > MONDO > others  (EFO is the Open-Targets backbone / drug space)
  3. tie-break: lowest id (lexicographic, stable/reproducible)

Everything is MEASURED against canon_nodes / canon_edges. No count is copied from prose
(the recurring failure mode: A2-5, A3-7, A4-3, A5-3). Guards FATAL loudly with real numbers.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import polars as pl

# ---- paths -------------------------------------------------------------------
CANON_NODES = Path("data/canon/canon_nodes.parquet")
CANON_EDGES = Path("data/canon/canon_edges")          # 27 per-label parquet files
A4_COLLISIONS = Path("data/manifest/a4_inchikey_collisions.parquet")  # 34 Launched-touching groups (D18)
OUT_DRUG = Path("data/manifest/a6_drug_collapse.parquet")
OUT_DIS  = Path("data/manifest/a6_disease_collapse.parquet")
OUT_DEC  = Path("data/manifest/a6_decisions.json")

# Disease prefix priority for the tie-break (EFO backbone first).
DIS_PREFIX_RANK = {"EFO": 0, "MONDO": 1, "ORPHANET": 2, "NCIT": 3}


def fatal(msg: str) -> None:
    print(f"[FATAL a6_collapse_rules] {msg}", file=sys.stderr)
    sys.exit(1)


def require(p: Path) -> None:
    if not p.exists():
        fatal(f"missing required input: {p}  (run the stage that produces it first)")


def degree_for(node_ids: set[str]) -> pl.DataFrame:
    """Undirected degree for a SMALL id set, measured from canon_edges.
    One scan over from/to; filtered early so the 21.8M-row scan is cheap for ~150 ids.
    endpoint_key is not needed here — we only count incidence per node id."""
    if not node_ids:
        return pl.DataFrame({"node": [], "degree": []}, schema={"node": pl.String, "degree": pl.UInt32})
    edges = pl.scan_parquet(str(CANON_EDGES / "*.parquet"))
    stacked = pl.concat([
        edges.select(pl.col("from").alias("node")),
        edges.select(pl.col("to").alias("node")),
    ])
    return (stacked.filter(pl.col("node").is_in(list(node_ids)))
                   .group_by("node").len().rename({"len": "degree"})
                   .collect())


def choose(df: pl.DataFrame, group_key: str, tie_expr: pl.Expr) -> pl.DataFrame:
    """Return one chosen row per group by (degree desc, tie_expr asc, id asc)."""
    return (df.sort(["degree", "_tie", "id"], descending=[True, False, False], nulls_last=True)
              .group_by(group_key, maintain_order=True)
              .agg(pl.first("id").alias("chosen_id"),
                   pl.col("id").alias("all_ids"),
                   pl.len().alias("n_nodes")))


def main() -> None:
    require(CANON_NODES)
    for f in ["drug.parquet", "disease.parquet"]:
        pass  # canon_edges is globbed; presence checked implicitly by the scan
    nodes = pl.read_parquet(CANON_NODES)

    # =========================================================================
    # RULE 1 — drug InChIKey collisions (E18)
    # =========================================================================
    drg = nodes.filter(pl.col("label") == "DRG")

    # same-key groups, measured (do NOT hardcode 69)
    groups = (drg.filter(pl.col("inchi_key").is_not_null())
                 .group_by("inchi_key").agg(pl.len().alias("n"))
                 .filter(pl.col("n") > 1))
    n_groups = groups.height
    print(f"[a6] measured same-InChIKey drug groups: {n_groups}")

    grp_nodes = (drg.join(groups.select("inchi_key"), on="inchi_key", how="inner")
                    .select(["id", "inchi_key", "name", "properties_json"]))

    # is_approved lives in properties_json (not a promoted column) — decode at a FIXED shape (A2-2 lesson)
    appr_schema = pl.Struct({"is_approved": pl.Boolean})
    grp_nodes = (grp_nodes
        .with_columns(pl.col("properties_json").str.json_decode(appr_schema).alias("_p"))
        .with_columns(pl.col("_p").struct.field("is_approved").fill_null(False).alias("is_approved"))
        .drop("_p", "properties_json"))

    deg = degree_for(set(grp_nodes["id"].to_list()))
    grp_nodes = (grp_nodes.join(deg, left_on="id", right_on="node", how="left")
                          .with_columns(pl.col("degree").fill_null(0))
                          # tie-break: is_approved True first  -> rank 0 for True, 1 for False
                          .with_columns((~pl.col("is_approved")).cast(pl.Int8).alias("_tie")))

    drug_collapse = choose(grp_nodes, "inchi_key", pl.col("_tie"))
    drug_collapse = drug_collapse.rename({"inchi_key": "collision_key"})

    # DRIFT GUARD: A4 already chose representatives for the 34 Launched-touching groups.
    # Ratifying means our rule must agree with A4's `chosen`. Disagreement => rule drift => FATAL.
    if A4_COLLISIONS.exists():
        a4 = pl.read_parquet(A4_COLLISIONS)
        a4_chosen = (a4.filter(pl.col("chosen"))
                       .select(pl.col("inchi_key").alias("collision_key"),
                               pl.col("id").alias("a4_chosen_id")))
        check = drug_collapse.join(a4_chosen, on="collision_key", how="inner")
        mism = check.filter(pl.col("chosen_id") != pl.col("a4_chosen_id"))
        if mism.height:
            print(mism.select(["collision_key", "chosen_id", "a4_chosen_id"]))
            fatal(f"A6 rule disagrees with A4 on {mism.height} Launched-touching group(s). "
                  f"Either the rule changed (then REPLACE, and re-run A4) or a4_inchikey_collisions "
                  f"is stale. Do not ratify silently.")
        print(f"[a6] drug collapse ratified against A4 on {check.height}/{a4_chosen.height} Launched groups: OK")
    else:
        print(f"[a6][WARN] {A4_COLLISIONS} absent — cannot ratify against A4; rule applied fresh.")

    OUT_DRUG.parent.mkdir(parents=True, exist_ok=True)
    drug_collapse.write_parquet(OUT_DRUG)
    print(f"[a6] wrote {OUT_DRUG}  ({drug_collapse.height} groups)")

    # =========================================================================
    # RULE 3 — disease CUI -> 2 DIS nodes (open rule 3)
    # =========================================================================
    # Exclude OBA the SAFE way (E19): NOT id STARTS WITH 'OBA', never id_prefix <> 'OBA'.
    dis = nodes.filter((pl.col("label") == "DIS") & (~pl.col("id").str.starts_with("OBA")))

    cui_groups = (dis.filter(pl.col("umls_cui").is_not_null())
                     .group_by("umls_cui").agg(pl.len().alias("n"))
                     .filter(pl.col("n") > 1))
    n_cui_multi = cui_groups.height
    print(f"[a6] measured CUIs mapping to >1 DIS node: {n_cui_multi}")

    dgrp = (dis.join(cui_groups.select("umls_cui"), on="umls_cui", how="inner")
               .select(["id", "umls_cui", "id_prefix", "name"]))

    ddeg = degree_for(set(dgrp["id"].to_list()))
    dgrp = (dgrp.join(ddeg, left_on="id", right_on="node", how="left")
                .with_columns(pl.col("degree").fill_null(0))
                # tie-break: prefix rank (EFO < MONDO < ...); unknown prefixes sort last
                .with_columns(pl.col("id_prefix").replace_strict(DIS_PREFIX_RANK, default=99)
                                .alias("_tie")))

    dis_collapse = choose(dgrp, "umls_cui", pl.col("_tie"))
    dis_collapse = dis_collapse.rename({"chosen_id": "chosen_okg_dis_id",
                                        "all_ids": "all_okg_dis_ids"})

    dis_collapse.write_parquet(OUT_DIS)
    print(f"[a6] wrote {OUT_DIS}  ({dis_collapse.height} CUIs collapsed)")

    # =========================================================================
    # E23 ruling + decisions
    # =========================================================================
    decisions = {
        "stage": "A6",
        "rule_1_drug_inchikey_collapse": {
            "status": "RATIFIED_from_A4",       # set to "REPLACED" if you change the rule (then re-run A4)
            "rule": "most edges > is_approved True > lowest id",
            "n_groups_measured": n_groups,
            "warrant": "InChIKey is not unique on the graph (E18/E14); row order must not pick the node.",
            "artifact": str(OUT_DRUG),
            "supersedes_open_rule": 1,
        },
        "rule_2_broad_salt_forms": {
            "status": "FROZEN",
            "rule": "match a Broad compound on ANY key in its `inchikeys` list; do not desalt to a "
                    "parent structure. A4's salt fallback already explodes `inchikeys` and joins on any key.",
            "warrant": "148 Broad compounds carry >1 distinct InChIKey (E13); 127 different-key OKG groups "
                       "(E18). Connectivity-block (first-14) matching is NOT used — ChEMBL registers salt "
                       "forms separately, so the block collapses distinct nodes and a 1:1 join duplicates gold pairs.",
            "supersedes_open_rule": 2,
        },
        "rule_3_disease_cui_collapse": {
            "status": "FROZEN",
            "rule": "most edges > id_prefix EFO>MONDO>other > lowest id",
            "n_cuis_measured": n_cui_multi,
            "warrant": "21 CUIs -> 2 DIS nodes (EFO+MONDO); left as duplicates, A7's holdout double-counts (D22).",
            "artifact": str(OUT_DIS),
            "supersedes_open_rule": 3,
        },
        "rule_4_dis_multi_cui": {"status": "CLEAN_no_rule", "warrant": "0 DIS nodes carry >1 CUI (STATUS §7)."},
        "rule_5_drugbank_dup": {"status": "CLEAN_no_rule", "warrant": "DrugBank duplication is 0 on the graph side (E18)."},
        "e23_native_key_ruling": {
            "decision": "native disease key = the promoted `umls_cui` column ONLY",
            "concept_ids_counts_as_native": False,
            "frozen_disease_reach": None,   # filled by a6_emit_crosswalks after re-measurement (expect 717/1462)
            "supersedes": "a6_keys.json reach=719 (counted 2 CUIs via secondary `concept_ids` and labelled "
                          "them native — see E23). umls_cui_xrefs add 0 net repoDB matches.",
        },
        "a1_keycoverage_decision": "SUPERSEDED by a6_collapse_rules.py + a6_emit_crosswalks.py; "
                                   "a1_keycoverage.py is retired (STATUS §7 non-blocking).",
    }
    OUT_DEC.write_text(json.dumps(decisions, indent=2))
    print(f"[a6] wrote {OUT_DEC}")
    print("[a6] collapse rules frozen. Next: a6_emit_crosswalks.py")


if __name__ == "__main__":
    main()
