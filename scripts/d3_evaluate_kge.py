#!/usr/bin/env python3
"""
ReTraceRx Phase D -- D3: Circularity-stratified, reachability-capped, filtered KGE evaluation.

For each held-out disease, rank candidate drugs, remove other known-true drugs for that disease
(FILTERED ranking), read off the target's rank, and aggregate Hits@1/3/10 + MRR by circularity stratum.

Two candidate modes (the pool the target is ranked within):
  --candidate-mode plus_true  (DEFAULT, capability eval): rank the true drug against the 1,565
     reachable-whitelist distractors WITH the true drug added to the pool. A reachable held-out
     drug outside the whitelist is still scorable -> populates Stratum A honestly. This is the
     metric of record for the KGE-vs-LLM-vs-agent ablation.
  --candidate-mode whitelist  (deployment eval): rank ONLY the fixed 1,565; a true drug outside it
     is unrecoverable and booked as a candidate-coverage ceiling (C_not_candidate). Report this
     ceiling alongside plus_true, do not substitute it for the capability number.

Corrections vs the how-to's D3 snippet (carried from D1/D2/Phase A):
  * Stratum C from a8_unreachable_drugs.parquet, not a non-existent ["unreachable_positives"] key.
  * Checksum compares DISTINCT unreachable drugs to a8's drug count (a8 counts drugs; a held-out
    positive is a pair -- one unreachable drug can span many pairs, esp. in drug_cold).
  * Guarded rank lookup (target absent -> None/miss, never a crash).
  * "Filtered" actually implemented (other known-true drugs for the disease removed first).

Terms (first use):
  filtered rank  target's position after removing every OTHER known-true drug for that disease.
  MRR            Mean Reciprocal Rank = mean of 1/rank (a miss contributes 0).
  Hits@k         fraction of queries whose target lands in the top k.
  Stratum A      graph-independent held-out positive (NOT a stripped DRG-DIS edge) -- PRIMARY.
  Stratum B      circular but reachable (was a stripped DRG-DIS edge) -- re-derivation, secondary.
  Stratum C      unreachable (no path to any DIS) -- a recall ceiling, excluded from the scored set.

Run:
  python scripts/d3_evaluate_kge.py --splits random disease_cold drug_cold
  python scripts/d3_evaluate_kge.py --splits random --candidate-mode whitelist   # deployment ceiling
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import polars as pl
import torch
from pykeen.triples import TriplesFactory
try:
    from pykeen.predict import predict_target
    _NEW = True
except Exception:
    _NEW = False

DATA      = Path("data")
HOLDOUT   = DATA / "holdout"
MANIFEST  = DATA / "manifest"
KGE_DATA  = Path("artifacts/kge_data")
OUT       = Path("artifacts/d3_eval")

REACHABLE_CSV = HOLDOUT / "candidate_whitelist_reachable.csv"     # 1,565 candidate drugs
UNREACH_PQ    = HOLDOUT / "a8_unreachable_drugs.parquet"          # graph-wide unreachable DRG
REACH_JSON    = HOLDOUT / "a8_reachability.json"                  # per-split unreachable DRUG counts
COLLAPSE_PQ   = MANIFEST / "a6_disease_collapse.parquet"
XWALK_PQ      = HOLDOUT / "disease_xwalk.parquet"
REMOVED_CSV   = HOLDOUT / "removed_edge_ids.csv"
SPLIT_POS     = lambda s: HOLDOUT / "splits" / f"{s}_holdout_positives.csv"


# --------------------------------------------------- disease-collapse remap (same rule as D1/D-D3)
def disease_remap() -> pl.DataFrame:
    collapse = pl.read_parquet(COLLAPSE_PQ)
    xwalk = pl.read_parquet(XWALK_PQ).select(okg_id="okg_dis_id", umls_cui="umls_cui")   # real col: okg_dis_id
    return (xwalk.join(collapse.select("umls_cui", "chosen_okg_dis_id"), on="umls_cui", how="inner")
                 .filter(pl.col("okg_id") != pl.col("chosen_okg_dis_id"))
                 .select(disease="okg_id", chosen="chosen_okg_dis_id").unique())

def apply_remap(df, remap, col):
    return (df.join(remap, left_on=col, right_on="disease", how="left")
              .with_columns(pl.coalesce(["chosen", col]).alias(col)).drop("chosen"))


# --------------------------------------------------- shared reference sets (split-independent)
def load_reference():
    reachable = set(pl.read_csv(REACHABLE_CSV)["okg_drug_id"].to_list())
    un = pl.read_parquet(UNREACH_PQ)
    id_col = next((c for c in ("okg_drug_id", "id", "drug_id", "okg_id") if c in un.columns), un.columns[0])
    unreachable = set(un[id_col].to_list())
    remap = disease_remap()
    removed = pl.read_csv(REMOVED_CSV)
    ind = (removed.filter((pl.col("label") == "DRG-DIS") & (pl.col("relation") == "INDICATION"))
                  .select(drug="from", disease="to").pipe(apply_remap, remap, "disease").unique())
    circular = set(zip(ind["drug"].to_list(), ind["disease"].to_list()))
    known: dict[str, set] = {}
    for d, dis in circular:
        known.setdefault(dis, set()).add(d)
    reach_counts = json.load(open(REACH_JSON)).get("holdout_unreachable_by_split", {})
    return reachable, unreachable, circular, known, remap, reach_counts


# --------------------------------------------------- metrics
def metrics(ranks):
    n = len(ranks)
    if n == 0:
        return dict(n=0, MRR=None, **{f"Hits@{k}": None for k in (1, 3, 10)})
    rr = [1.0 / r for r in ranks if r is not None]
    return dict(n=n, MRR=round(sum(rr) / n, 4),
                **{f"Hits@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
                   for k in (1, 3, 10)})


# --------------------------------------------------- per-split evaluation
def evaluate_split(split, reachable, unreachable, circular, known, remap, reach_counts,
                   device, candidate_mode):
    tf = TriplesFactory.from_path_binary(str(KGE_DATA / split / "train_factory"))
    IND = json.load(open(KGE_DATA / split / "d1_meta.json"))["ind_relation"]
    assert IND in tf.relation_to_id, f"{IND!r} not in model relation vocab"
    model = torch.load(f"artifacts/kge_rotate_{split}/trained_model.pkl", weights_only=False).to(device)
    model.eval()

    pos = (pl.read_csv(SPLIT_POS(split)).select(drug="okg_drug_id", disease="okg_disease_id")
             .pipe(apply_remap, remap, "disease").unique())
    for d, dis in zip(pos["drug"].to_list(), pos["disease"].to_list()):
        known.setdefault(dis, set()).add(d)

    by_dis = {}
    for d, dis in zip(pos["drug"].to_list(), pos["disease"].to_list()):
        by_dis.setdefault(dis, []).append(d)

    records = []
    for dis, drugs in by_dis.items():
        base_ranking = None                                    # cached per disease
        for true_drug in drugs:
            # Stratum C (reachability ceiling) always takes precedence
            if true_drug in unreachable:
                records.append((split, dis, true_drug, "C_unreachable", None)); continue
            # deployment mode only: a reachable non-candidate cannot be ranked
            if candidate_mode == "whitelist" and true_drug not in reachable:
                records.append((split, dis, true_drug, "C_not_candidate", None)); continue

            stratum = "B_circular" if (true_drug, dis) in circular else "A_independent"

            if base_ranking is None:
                if _NEW:
                    df = predict_target(model=model, tail=dis, relation=IND, triples_factory=tf).df
                    col = "head_label" if "head_label" in df.columns else "head_id"
                else:
                    df = model.get_head_prediction_df(relation_label=IND, tail_label=dis, triples_factory=tf)
                    col = "head_label"
                df = df.sort_values("score", ascending=False)
                base_ranking = df[[col, "score"]].values.tolist()   # [[drug, score], ...] best-first

            pool = (reachable | {true_drug}) if candidate_mode == "plus_true" else reachable
            ranked = [d for d, _ in base_ranking if d in pool]

            # FILTERED: drop OTHER known-true drugs for this disease, keep the target
            others = known.get(dis, set()) - {true_drug}
            filtered = [d for d in ranked if d == true_drug or d not in others]
            rank = (filtered.index(true_drug) + 1) if true_drug in filtered else None
            records.append((split, dis, true_drug, stratum, rank))

    rec = pl.DataFrame(records, schema=["split", "disease", "drug", "stratum", "rank"], orient="row")

    def ranks_for(s): return rec.filter(pl.col("stratum") == s)["rank"].to_list()
    A, B = ranks_for("A_independent"), ranks_for("B_circular")

    # ceiling accounting -- checksum on DISTINCT DRUGS (a8 counts drugs, a positive is a pair)
    C_un_pairs = rec.filter(pl.col("stratum") == "C_unreachable").height
    C_un_drugs = rec.filter(pl.col("stratum") == "C_unreachable")["drug"].n_unique()
    C_nc_pairs = rec.filter(pl.col("stratum") == "C_not_candidate").height     # 0 in plus_true mode
    scored = A + B
    full   = scored + [None] * (C_un_pairs + C_nc_pairs)

    n = rec.height
    a8 = reach_counts.get(split)
    checksum_ok = (a8 is None or C_un_drugs == a8)
    summary = dict(
        split=split, candidate_mode=candidate_mode, ind_relation=IND, n_positives=n,
        stratum_A_independent=metrics(A), stratum_B_circular=metrics(B),
        reachable_scored_AB=metrics(scored), full_with_ceiling_as_miss=metrics(full),
        ceiling=dict(
            C_unreachable_pairs=C_un_pairs, C_unreachable_drugs=C_un_drugs,
            C_not_candidate_pairs=C_nc_pairs,
            a8_unreachable_drug_count=a8, checksum_ok=checksum_ok,
            max_achievable_recall=round((n - C_un_pairs - C_nc_pairs) / n, 4) if n else None,
            max_recall_reachability_only=round((n - C_un_pairs) / n, 4) if n else None,
        ),
        note="Stratum A is PRIMARY. plus_true = capability eval (record); whitelist = deployment "
             "ceiling. C_unreachable = reachability ceiling; C_not_candidate only appears in "
             "whitelist mode (reachable drug absent from the 1,565).",
    )
    if not checksum_ok:
        print(f"  [!] [{split}] Stratum C distinct drugs ({C_un_drugs}) != a8 count ({a8}) "
              f"-- investigate before trusting.")
    return summary, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["random", "disease_cold"],
                    choices=["random", "drug_cold", "disease_cold"])
    ap.add_argument("--candidate-mode", choices=["plus_true", "whitelist"], default="plus_true",
                    help="plus_true = capability eval (default); whitelist = fixed-shortlist deployment eval")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    ref = load_reference()
    for split in args.splits:
        print(f"\n=== D3 eval: {split}  (candidate_mode={args.candidate_mode}) ===")
        summary, rec = evaluate_split(split, *ref, device, args.candidate_mode)
        tag = args.candidate_mode
        json.dump(summary, open(OUT / f"d3_metrics_{split}_{tag}.json", "w"), indent=2)
        rec.write_csv(OUT / f"d3_perquery_{split}_{tag}.csv")
        a, c = summary["stratum_A_independent"], summary["ceiling"]
        print(f"[{split}] positives={summary['n_positives']}  "
              f"A(primary) n={a['n']} Hits@10={a['Hits@10']} MRR={a['MRR']}  |  "
              f"B n={summary['stratum_B_circular']['n']}  "
              f"C_unreach={c['C_unreachable_pairs']}p/{c['C_unreachable_drugs']}d "
              f"(a8 {c['a8_unreachable_drug_count']}d, ok={c['checksum_ok']})  "
              f"C_notcand={c['C_not_candidate_pairs']}  max_recall={c['max_achievable_recall']}")

    print(f"\nD3 complete -> {OUT}. Per-query CSVs feed the D4 shared harness.")


if __name__ == "__main__":
    main()
