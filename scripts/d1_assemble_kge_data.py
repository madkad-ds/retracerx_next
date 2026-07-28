#!/usr/bin/env python3
"""
ReTraceRx Phase D -- D1: Assemble the KGE data (compensating for whole-label removal).

For each holdout split it builds a PyKEEN TriplesFactory pair:
    train = mechanism triples  UNION  train-split indication triples
    test  = held-out (test-split) indication triples

Correctness safeguards beyond the how-to's D1 sketch (each maps to a Phase A record fact):
  1. The composite indication relation string is DERIVED from kge_triples.tsv, not
     hardcoded -- the DRG-DIS vs DRG_DIS casing is unverified (Phase A pitfall #3).
  2. The A6 disease-collapse map is applied to reconstructed indication edges BEFORE
     the split, so a held-out pair cannot re-enter train under a non-chosen (alternate)
     disease id (leakage; the class A7 gate G5 guards against).
  3. Split-aware exclusion: random -> exclude exact test pairs; drug_cold -> exclude every
     indication edge whose DRUG is held out; disease_cold -> whose DISEASE is held out.
     Done with an anti-join / set filter, not a per-row Python UDF.
  4. Explicit no-leakage assertion + an entity-vocabulary report cross-checked against
     a8_reachability.json (the 23/16/20 unreachable positives).

Run:  python d1_assemble_kge_data.py --splits random disease_cold
Outputs (per split) under artifacts/kge_data/<split>/:
    train.tsv, train_factory/, test_factory/, d1_meta.json
"""

from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
import polars as pl
from pykeen.triples import TriplesFactory

# ----------------------------------------------------------------------------- paths
DATA      = Path("data")
HOLDOUT   = DATA / "holdout"
MANIFEST  = DATA / "manifest"
OUT       = Path("artifacts/kge_data")

KGE_TSV      = DATA / "kge_triples.tsv"                       # 21,728,106 mechanism triples
REMOVED_CSV  = HOLDOUT / "removed_edge_ids.csv"              # from,to,label,relation,endpoint_key
COLLAPSE_PQ  = MANIFEST / "a6_disease_collapse.parquet"          # umls_cui -> chosen_okg_dis_id
XWALK_PQ     = HOLDOUT / "disease_xwalk.parquet"             # okg_id, umls_cui, ...
REACH_JSON   = HOLDOUT / "a8_reachability.json"              # unreachable_positives[<split>]
# per-split held-out positives; VERIFY this path against your frozen bundle:
SPLIT_POS    = lambda s: HOLDOUT / "splits" / f"{s}_holdout_positives.csv"  # okg_drug_id, okg_disease_id

SEP = "__"   # composite separator label__relation; asserted below


# ------------------------------------------------------- Step 1: derive the IND relation
def derive_indication_relation(mech_relations: set[str]) -> str:
    """Reconstruct the exact composite string for DRG-DIS INDICATION from the naming
    convention actually used by the surviving DRG-* relations in kge_triples.tsv.
    The label was fully removed, so we cannot read the string off the file directly --
    we copy the *convention* (hyphen-vs-underscore + separator) from a sibling label."""
    assert any(SEP in r for r in mech_relations), \
        f"No '{SEP}' separator found in mech relations -- composite convention differs; inspect the file."
    # label part = substring before the first separator
    drg_labels = {r.split(SEP, 1)[0] for r in mech_relations if r.split(SEP, 1)[0].startswith("DRG")}
    assert drg_labels, "No surviving DRG-* relation to learn the label convention from."
    sample = sorted(drg_labels)[0]                 # e.g. 'DRG-DRG' or 'DRG_DRG'
    joiner = sample[3]                             # the char between the two 3-letter codes
    assert joiner in "-_", f"Unexpected joiner char {joiner!r} in label {sample!r}."
    ind = f"DRG{joiner}DIS{SEP}INDICATION"         # relation part 'INDICATION' matches removed.csv
    assert ind not in mech_relations, \
        f"{ind} already present in mech -- DRG-DIS was NOT fully removed; stop and re-check A7."
    print(f"[IND] derived indication relation = {ind!r}  (learned joiner {joiner!r} from {sample!r})")
    return ind


# ------------------------------------------------ Step 2: okg_dis_id -> chosen_okg_dis_id
def build_disease_remap() -> pl.DataFrame:
    """Map any (possibly non-chosen) OKG disease id to its chosen id via the A6 collapse.
    Built as xwalk(okg_id -> umls_cui) JOIN collapse(umls_cui -> chosen). Non-collapsed
    ids simply never appear here and default to identity at apply time."""
    collapse = pl.read_parquet(COLLAPSE_PQ)
    assert "chosen_okg_dis_id" in collapse.columns and "umls_cui" in collapse.columns, \
        f"a6_disease_collapse columns are {collapse.columns}; expected umls_cui + chosen_okg_dis_id."
    xwalk = pl.read_parquet(XWALK_PQ).select(
        okg_id="okg_dis_id", umls_cui="umls_cui")      # rename here if your xwalk uses other names
    remap = (xwalk.join(collapse.select("umls_cui", "chosen_okg_dis_id"), on="umls_cui", how="inner")
                  .filter(pl.col("okg_id") != pl.col("chosen_okg_dis_id"))   # only true rewrites
                  .select(disease="okg_id", chosen="chosen_okg_dis_id")
                  .unique())
    print(f"[collapse] {remap.height} alternate disease ids will be rewritten to their chosen id")
    return remap


def apply_remap(df: pl.DataFrame, remap: pl.DataFrame, col: str) -> pl.DataFrame:
    return (df.join(remap, left_on=col, right_on="disease", how="left")
              .with_columns(pl.coalesce(["chosen", col]).alias(col))
              .drop("chosen"))


# --------------------------------------------------------- Step 3: per-split factory pair
def kge_factory_for_split(split: str, mech: pl.DataFrame, ind_edges: pl.DataFrame,
                          remap: pl.DataFrame, IND: str, reach: dict) -> tuple[TriplesFactory, TriplesFactory]:
    # 3a. test = frozen held-out positives, disease ids normalized to the collapsed id space
    test = (pl.read_csv(SPLIT_POS(split))
              .select(drug="okg_drug_id", disease="okg_disease_id")
              .pipe(apply_remap, remap, "disease")
              .unique())
    heldout_pairs    = set(zip(test["drug"], test["disease"]))
    heldout_drugs    = set(test["drug"])
    heldout_diseases = set(test["disease"])

    # 3b. split-aware exclusion of training indications (see module docstring #3)
    if split == "random":
        train_ind = ind_edges.filter(
            ~pl.struct(["drug", "disease"]).is_in(
                [{"drug": d, "disease": s} for d, s in heldout_pairs]))
        # faster, equivalent anti-join form for large sets:
        train_ind = ind_edges.join(
            test.select("drug", "disease"), on=["drug", "disease"], how="anti")
    elif split == "drug_cold":
        train_ind = ind_edges.filter(~pl.col("drug").is_in(list(heldout_drugs)))
    elif split == "disease_cold":
        train_ind = ind_edges.filter(~pl.col("disease").is_in(list(heldout_diseases)))
    else:
        raise ValueError(f"unknown split {split!r}")

    train_ind = train_ind.unique(subset=["drug", "disease"])

    # 3c. LEAKAGE ASSERTION -- no held-out pair may survive in train
    leaked = set(zip(train_ind["drug"], train_ind["disease"])) & heldout_pairs
    assert not leaked, f"LEAKAGE: {len(leaked)} held-out pairs present in train for split {split!r}"

    # 3d. assemble train = mechanism UNION train indications, as composite (h, r, t)
    train_ind_tri = train_ind.select(h="drug", r=pl.lit(IND), t="disease")
    train = pl.concat([mech, train_ind_tri]).unique()

    # 3e. build the training factory from a temp TSV (avoids a 21.7M-row str numpy array)
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
        tmp = f.name
    train.write_csv(tmp, separator="\t", include_header=False)
    tf_tr = TriplesFactory.from_path(tmp, create_inverse_triples=True)   # D-A8-2: inverses here
    os.unlink(tmp)

    # 3f. test factory -- only pairs whose entities exist in the transductive train vocab
    ents = set(tf_tr.entity_to_id)
    scoreable = test.filter(pl.col("drug").is_in(list(ents)) & pl.col("disease").is_in(list(ents)))
    dropped   = test.height - scoreable.height
    by_split = reach.get("holdout_unreachable_by_split", {})
    _v = by_split.get(split, 0)
    unreach_expected = len(_v) if isinstance(_v, list) else int(_v)   
    print(f"[{split}] test positives={test.height} scoreable={scoreable.height} "
          f"unscoreable(missing entity)={dropped}  (a8 graph-unreachable={unreach_expected})")
    

    test_tri = scoreable.select(h="drug", r=pl.lit(IND), t="disease").to_numpy().astype(str)
    tf_te = TriplesFactory.from_labeled_triples(
        test_tri, entity_to_id=tf_tr.entity_to_id, relation_to_id=tf_tr.relation_to_id)

    assert IND in tf_tr.relation_to_id, f"{IND} missing from train relation vocab"
    meta = dict(split=split, ind_relation=IND,
                n_train_triples=train.height, n_train_indications=train_ind.height,
                n_test_positives=test.height, n_test_scoreable=scoreable.height,
                n_test_unscoreable=dropped, a8_unreachable=unreach_expected,
                unscoreable_pairs=list(zip(
                    test.join(scoreable, on=["drug", "disease"], how="anti")["drug"],
                    test.join(scoreable, on=["drug", "disease"], how="anti")["disease"])))
    return tf_tr, tf_te, meta


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["random", "disease_cold"],
                    choices=["random", "drug_cold", "disease_cold"])
    args = ap.parse_args()

    print("Loading mechanism triples (21.7M)...")
    mech = pl.read_csv(KGE_TSV, separator="\t", has_header=False, new_columns=["h", "r", "t"])
    mech_relations = set(mech["r"].unique().to_list())
    IND = derive_indication_relation(mech_relations)

    removed = pl.read_csv(REMOVED_CSV)
    for c in ("from", "to", "label", "relation"):
        assert c in removed.columns, f"removed_edge_ids.csv missing column {c!r}: {removed.columns}"
    ind_edges = (removed
                 .filter((pl.col("label") == "DRG-DIS") & (pl.col("relation") == "INDICATION"))
                 .select(drug="from", disease="to"))
    print(f"[ind] {ind_edges.height} raw DRG-DIS/INDICATION edges recovered from removed_edge_ids")

    remap = build_disease_remap()
    ind_edges = apply_remap(ind_edges, remap, "disease").unique()
    print(f"[ind] {ind_edges.height} indication edges after collapse + dedupe")

    reach = json.load(open(REACH_JSON)) if REACH_JSON.exists() else {}

    for split in args.splits:
        print(f"\n=== split: {split} ===")
        tf_tr, tf_te, meta = kge_factory_for_split(split, mech, ind_edges, remap, IND, reach)
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        tf_tr.to_path_binary(str(d / "train_factory"))
        tf_te.to_path_binary(str(d / "test_factory"))
        json.dump(meta, open(d / "d1_meta.json", "w"), indent=2)
        print(f"[{split}] saved -> {d}  train_triples={meta['n_train_triples']:,} "
              f"train_ind={meta['n_train_indications']:,} test={meta['n_test_scoreable']}")

    print("\nD1 complete. Feed artifacts/kge_data/<split>/{train_factory,test_factory} to D2.")


if __name__ == "__main__":
    main()
