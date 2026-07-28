#!/usr/bin/env python3
"""
a6_ratify_drug_collapse.py  —  Deep ratification of the drug InChIKey collapse map.

Supersedes the shallow `19/49 ... OK` check in a6_collapse_rules.py.

WHAT "RATIFIED" MEANS HERE (D18, operationalized):
  The 1,700-node candidate whitelist that A4 shipped must be *derivable from the frozen
  collapse map*. We test the artifact that shipped, not A4's internal `chosen` flags
  (their denominator is inconsistent: STATUS says 34, the map guard saw 49).

SCOPE (corrected — see the A6 discussion):
  * The collapse map covers ALL same-InChIKey DRG groups (superset; a subset can silently
    miss a key). E18's "69" is LCC AND name-duplicate scoped and would drop same-key/
    different-name collisions — it is NOT the right population.
  * The drug InChIKey collapse is consumed by ONE join only: Broad -> OKG (Launched).
    repoDB reaches drugs via DrugBank (1:1, no collapse). So RATIFICATION is defined over
    the LAUNCHED-TOUCHING subset, measured directly, not over 69 or 75.

LCC-AWARENESS (E16/E17):
  The 982 LCC-excluded drugs have internal degrees up to 63. "most edges" alone can pick an
  island node the whitelist drops and inference excludes. A valid representative must be an
  LCC member whenever the group contains one.

OUTCOMES:
  MAP_DEFECT     -> an island node was chosen over an available LCC member. FATAL. Fix the rule
                   in a6_collapse_rules.py and re-emit the map. Do NOT re-run A4 (it would
                   propagate the defect).
  WHITELIST_DRIFT-> map is sound but the shipped whitelist disagrees. LAST RESORT: re-run
                   a4_candidate_universe.py against the frozen map, then re-verify.
  RATIFIED       -> whitelist reflects the map on every Launched-touching group. Flip
                   collapse_rule_status -> RATIFIED with the full denominator.

Usage:
  python scripts/a6_ratify_drug_collapse.py                 # check only; FATAL on drift
  python scripts/a6_ratify_drug_collapse.py --allow-a4-rerun  # permit the last-resort re-run
"""
from __future__ import annotations
import argparse, json, subprocess, sys, hashlib
from pathlib import Path
import polars as pl

CANON_NODES = Path("data/canon/canon_nodes.parquet")
CANON_EDGES = Path("data/canon/canon_edges")
LCC_EXCL    = Path("data/manifest/a3_lcc_excluded.parquet")          # id, label, degree, component_id, ...
BROAD       = Path("data/canon/broad_compounds.parquet")
A4_COLL     = Path("data/manifest/a4_inchikey_collisions.parquet")   # inchi_key, id, name, is_approved, degree, chosen
MAP         = Path("data/manifest/a6_drug_collapse.parquet")         # collision_key, chosen_id, all_ids, n_nodes
WHITELIST   = Path("data/canon/candidate_whitelist.parquet")         # okg_drug_id, inchi_key, pert_iname, ...
A6_DEC      = Path("data/manifest/a6_decisions.json")

# The command run as the LAST RESORT. A4 MUST consume the frozen map for this to be meaningful.
A4_RERUN_CMD = ["python", "scripts/a4_candidate_universe.py",
                "--collapse-map", str(MAP)]


def fatal(msg: str) -> None:
    print(f"[FATAL a6_ratify] {msg}", file=sys.stderr)
    sys.exit(1)


def require(p: Path) -> None:
    if not p.exists():
        fatal(f"missing required input: {p}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def degree_for(ids: set[str]) -> pl.DataFrame:
    if not ids:
        return pl.DataFrame(schema={"node": pl.String, "degree": pl.UInt32})
    edges = pl.scan_parquet(str(CANON_EDGES / "*.parquet"))
    return (pl.concat([edges.select(pl.col("from").alias("node")),
                       edges.select(pl.col("to").alias("node"))])
              .filter(pl.col("node").is_in(list(ids)))
              .group_by("node").len().rename({"len": "degree"}).collect())


def is_approved_map(drg: pl.DataFrame, ids: set[str]) -> pl.DataFrame:
    schema = pl.Struct({"is_approved": pl.Boolean})
    return (drg.filter(pl.col("id").is_in(list(ids))).select(["id", "properties_json"])
               .with_columns(pl.col("properties_json").str.json_decode(schema).alias("_p"))
               .with_columns(pl.col("_p").struct.field("is_approved").fill_null(False).alias("is_approved"))
               .select(["id", "is_approved"]))


def lcc_aware_representative(group_nodes: pl.DataFrame) -> str:
    """Choose one id per the LCC-aware rule:
       lcc_member True first  >  most edges  >  is_approved True  >  lowest id."""
    ranked = group_nodes.sort(
        by=["lcc_member", "degree", "is_approved", "id"],
        descending=[True, True, True, False],
        nulls_last=True,
    )
    return ranked["id"][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-a4-rerun", action="store_true",
                    help="permit the last-resort re-run of a4_candidate_universe.py on WHITELIST_DRIFT")
    args = ap.parse_args()

    for p in (CANON_NODES, LCC_EXCL, BROAD, MAP, WHITELIST):
        require(p)

    nodes = pl.read_parquet(CANON_NODES)
    drg = nodes.filter(pl.col("label") == "DRG")
    lcc_excluded_ids = set(pl.read_parquet(LCC_EXCL)["id"].to_list())
    cmap = pl.read_parquet(MAP)
    whitelist = pl.read_parquet(WHITELIST)

    # ------------------------------------------------------------------ #
    # 1. Reconcile the collision population (75 vs 69) with an LCC partition.
    # ------------------------------------------------------------------ #
    groups = (drg.filter(pl.col("inchi_key").is_not_null())
                 .group_by("inchi_key").agg(ids=pl.col("id"), n=pl.len())
                 .filter(pl.col("n") > 1))
    n_all = groups.height

    def group_lcc_class(id_list: list[str]) -> str:
        n_in = sum(i not in lcc_excluded_ids for i in id_list)
        if n_in == len(id_list):
            return "all_in_lcc"
        if n_in == 0:
            return "all_excluded"
        return "spans_boundary"

    groups = groups.with_columns(
        pl.col("ids").map_elements(group_lcc_class, return_dtype=pl.String).alias("lcc_class"))
    part = groups.group_by("lcc_class").len().sort("lcc_class")
    print(f"[ratify] collision groups (all same-key DRG): {n_all}")
    print(part)
    print(f"[ratify] E18's 69 is LCC AND name-duplicate scoped; {n_all} is the full key population "
          f"(superset — the safe scope). The delta is recorded, not treated as drift.")

    # ------------------------------------------------------------------ #
    # 2. Operative set = Launched-touching collision keys (the only consumed set).
    # ------------------------------------------------------------------ #
    broad = pl.read_parquet(BROAD)
    launched = broad.filter(pl.col("clinical_phase") == "Launched")
    launched_keys = set(launched.filter(pl.col("inchi_key").is_not_null())["inchi_key"].to_list())
    if "inchikeys" in launched.columns:
        salt = launched.filter(pl.col("inchi_key").is_null()).select("inchikeys").explode("inchikeys")
        launched_keys |= set(salt.drop_nulls("inchikeys")["inchikeys"].to_list())

    operative = groups.filter(pl.col("inchi_key").is_in(list(launched_keys)))
    op_keys = set(operative["inchi_key"].to_list())
    print(f"[ratify] Launched-touching collision groups (the ratification denominator): {len(op_keys)}")
    if not op_keys:
        print("[ratify] no Launched-touching collisions — drug collapse is vacuously ratified.")

    # ------------------------------------------------------------------ #
    # 3. Re-derive the LCC-aware representative for every operative group.
    #    Then check the MAP for defects, and the WHITELIST for drift.
    # ------------------------------------------------------------------ #
    op_node_ids = {i for row in operative["ids"].to_list() for i in row}
    deg = degree_for(op_node_ids)
    appr = is_approved_map(drg, op_node_ids)
    op_nodes = (operative.explode("ids").rename({"ids": "id"}).select(["inchi_key", "id"])
                .join(deg, left_on="id", right_on="node", how="left")
                .with_columns(pl.col("degree").fill_null(0))
                .join(appr, on="id", how="left")
                .with_columns(pl.col("is_approved").fill_null(False))
                .with_columns((~pl.col("id").is_in(list(lcc_excluded_ids))).alias("lcc_member")))

    map_defects, whitelist_drift, agree = [], [], []
    wl_by_key = (whitelist.select(["inchi_key", "okg_drug_id"])
                          .group_by("inchi_key").agg(pl.col("okg_drug_id").alias("wl_ids")))
    wl_lookup = {r["inchi_key"]: set(r["wl_ids"]) for r in wl_by_key.iter_rows(named=True)}
    map_lookup = {r["collision_key"]: r["chosen_id"] for r in cmap.iter_rows(named=True)}

    for key in sorted(op_keys):
        gnodes = op_nodes.filter(pl.col("inchi_key") == key)
        want = lcc_aware_representative(gnodes)                 # the correct representative
        mapped = map_lookup.get(key)                           # what the frozen map chose
        wl_ids = wl_lookup.get(key, set())                     # what the whitelist actually shipped

        # (a) MAP_DEFECT: map chose an island node while an LCC member exists in the group.
        has_lcc = bool(gnodes.filter(pl.col("lcc_member"))["id"].len())
        if mapped is None:
            map_defects.append((key, "missing_from_map", want))
        elif has_lcc and (mapped in lcc_excluded_ids):
            map_defects.append((key, f"chose_island_{mapped}", want))
        elif mapped != want:
            # map disagrees with the LCC-aware rule but is not an island choice — treat as defect
            # (the map must be regenerated so it matches the frozen rule).
            map_defects.append((key, f"map_{mapped}_vs_rule_{want}", want))

        # (b) WHITELIST_DRIFT: the shipped whitelist points at a node other than the map's choice.
        if mapped is not None and wl_ids and (wl_ids != {mapped}):
            whitelist_drift.append((key, sorted(wl_ids), mapped))
        elif mapped is not None and wl_ids == {mapped}:
            agree.append(key)

    print(f"\n[ratify] agree (whitelist reflects map): {len(agree)}/{len(op_keys)}")

    # ------------------------------------------------------------------ #
    # 4. MAP_DEFECT must be fixed upstream — never resolved by an A4 re-run.
    # ------------------------------------------------------------------ #
    if map_defects:
        print("[ratify] MAP_DEFECT groups (key, problem, correct_representative):")
        for row in map_defects:
            print("   ", row)
        fatal(f"{len(map_defects)} map defect(s). The frozen collapse map disagrees with the "
              f"LCC-aware rule (or chose an island node). Fix the rule in a6_collapse_rules.py "
              f"(rank lcc_member first), re-emit a6_drug_collapse.parquet, then re-run this. "
              f"Do NOT re-run A4 — it would propagate the defect.")

    # ------------------------------------------------------------------ #
    # 5. WHITELIST_DRIFT -> last-resort A4 re-run (only if the map is clean).
    # ------------------------------------------------------------------ #
    if whitelist_drift:
        print("[ratify] WHITELIST_DRIFT groups (key, whitelist_nodes, map_choice):")
        for row in whitelist_drift:
            print("   ", row)
        if not args.allow_a4_rerun:
            fatal(f"{len(whitelist_drift)} group(s): whitelist does not reflect the frozen map. "
                  f"Re-run with --allow-a4-rerun to invoke the last-resort A4 rebuild, or "
                  f"regenerate the whitelist manually from a6_drug_collapse.parquet.")

        print(f"[ratify] LAST RESORT: re-running A4 against the frozen map:\n   {' '.join(A4_RERUN_CMD)}")
        before = sha256(WHITELIST)
        res = subprocess.run(A4_RERUN_CMD, capture_output=True, text=True)
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        if res.returncode != 0:
            fatal("A4 re-run exited non-zero. Whitelist unchanged; not ratified.")
        after = sha256(WHITELIST)
        if after == before:
            fatal("A4 re-run left candidate_whitelist.parquet byte-identical. Either A4 does not "
                  "accept --collapse-map (so the re-run was a no-op) or it ignores the map. Wire A4 "
                  "to consume data/manifest/a6_drug_collapse.parquet, then re-run this check.")

        # Re-verify the drift is gone.
        whitelist = pl.read_parquet(WHITELIST)
        wl_by_key = (whitelist.select(["inchi_key", "okg_drug_id"])
                              .group_by("inchi_key").agg(pl.col("okg_drug_id").alias("wl_ids")))
        wl_lookup = {r["inchi_key"]: set(r["wl_ids"]) for r in wl_by_key.iter_rows(named=True)}
        still = [k for k in whitelist_drift
                 if wl_lookup.get(k[0], set()) != {map_lookup[k[0]]}]
        if still:
            print("[ratify] still drifted after re-run:", still)
            fatal("A4 re-run did not reconcile every group. Not ratified.")
        print("[ratify] A4 re-run reconciled all drift. Whitelist now reflects the map.")

    # ------------------------------------------------------------------ #
    # 6. RATIFIED — record it with the FULL denominator.
    # ------------------------------------------------------------------ #
    if A6_DEC.exists():
        dec = json.loads(A6_DEC.read_text())
        r1 = dec.setdefault("rule_1_drug_inchikey_collapse", {})
        r1["status"] = "RATIFIED"
        r1["ratification"] = {
            "definition": "candidate_whitelist is derivable from a6_drug_collapse on every "
                          "Launched-touching collision group",
            "operative_groups": len(op_keys),
            "agree": len(agree),
            "a4_rerun_used": bool(whitelist_drift),
            "collision_population_full": n_all,
            "collision_population_note": "full same-key DRG population (superset); E18's 69 is "
                                         "LCC AND name-duplicate scoped and is not the map population",
            "lcc_aware_rule": "lcc_member > most edges > is_approved > lowest id",
        }
        A6_DEC.write_text(json.dumps(dec, indent=2))
        print(f"[ratify] wrote RATIFIED status + full denominator to {A6_DEC}")

    print(f"\n[ratify] RATIFIED: {len(op_keys)} Launched-touching group(s), whitelist reflects the map.")


if __name__ == "__main__":
    main()
