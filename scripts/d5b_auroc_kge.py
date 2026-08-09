#!/usr/bin/env python3
"""
ReTraceRx Phase D -- D5(b): Discrimination with real negatives (AUROC / AUPRC) for the KGE baseline.

Scores held-out POSITIVES (label 1) and repoDB-derived NEGATIVES (label 0, real trial failures) with
the D2 KGE model on (okg_drug, INDICATION, okg_disease), then computes AUROC and AUPRC per split.

Hygiene (matches D1/D3/D4):
  * Negatives come from the FROZEN holdout bundle (holdout_negatives.csv) -- already okg-keyed,
    collapse-remapped, and split-scoped. No drugbank->okg resolution, no cross-split mixing.
  * Positives are HELD-OUT (never trained on) -> AUROC is not inflated by trained-on edges.
  * Disease ids re-collapsed (idempotent) so both sides share the model's id space.
  * Pairs whose drug/disease are absent from the model vocab are DROPPED and COUNTED (coverage caveat,
    same discipline as the ranking ceilings) -- never silently lost.
  * Reported both overall (A+B positives) and Stratum-A-only positives vs the shared negatives.
  * Score = model.score_hrt on the FORWARD indication relation (trained with inverse triples, D-D5).

Run:
  python scripts/d5b_auroc_kge.py --splits random disease_cold drug_cold
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from pykeen.triples import TriplesFactory

from d3_evaluate_kge import load_reference, apply_remap, KGE_DATA, HOLDOUT

OUT = Path("artifacts/d5_discrimination")

SPLIT_POS = lambda s: HOLDOUT / "splits" / f"{s}_holdout_positives.csv"
def split_neg_path(split):
    p = HOLDOUT / "splits" / f"{split}_holdout_negatives.csv"
    if p.exists():
        return p
    if split == "disease_cold" and (HOLDOUT / "holdout_negatives.csv").exists():
        return HOLDOUT / "holdout_negatives.csv"            # headline file is disease_cold
    raise FileNotFoundError(
        f"No negatives for split {split!r}. Discrimination needs each split's OWN negatives "
        f"(mixing splits voids it). Looked for {p} and the headline holdout_negatives.csv.")


def score_pairs(df, model, e2i, rid, device):
    """df has label columns drug, disease. Returns (scores ndarray, kept df, n_dropped)."""
    keep = df.filter(pl.col("drug").is_in(list(e2i)) & pl.col("disease").is_in(list(e2i)))
    dropped = df.height - keep.height
    if keep.height == 0:
        return np.array([]), keep, dropped
    h = [e2i[d] for d in keep["drug"].to_list()]
    t = [e2i[x] for x in keep["disease"].to_list()]
    hrt = torch.tensor([[hi, rid, ti] for hi, ti in zip(h, t)], dtype=torch.long, device=device)
    with torch.no_grad():
        s = model.score_hrt(hrt).detach().cpu().numpy().ravel()
    return s, keep, dropped


def auc_ci(y, s, n_boot=2000, seed=0):
    """Bootstrap 95% CI for AUROC and AUPRC (per-pair resample)."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    roc, prc = [], []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:      # need both classes in the resample
            continue
        roc.append(roc_auc_score(y[b], s[b])); prc.append(average_precision_score(y[b], s[b]))
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)] if a else None
    return ci(roc), ci(prc)


def evaluate_split(split, ref, device):
    reachable, unreachable, circular, known, remap, reach_counts = ref
    tf = TriplesFactory.from_path_binary(str(KGE_DATA / split / "train_factory"))
    IND = json.load(open(KGE_DATA / split / "d1_meta.json"))["ind_relation"]
    e2i, r2i = tf.entity_to_id, tf.relation_to_id
    assert IND in r2i, f"{IND!r} not in relation vocab"
    rid = r2i[IND]
    model = torch.load(f"artifacts/kge_rotate_{split}/trained_model.pkl", weights_only=False).to(device)
    model.eval()

    pos = (pl.read_csv(SPLIT_POS(split)).select(drug="okg_drug_id", disease="okg_disease_id")
             .pipe(apply_remap, remap, "disease").unique())
    neg = (pl.read_csv(split_neg_path(split)).select(drug="okg_drug_id", disease="okg_disease_id")
             .pipe(apply_remap, remap, "disease").unique())

    # tag positives with stratum (A independent, B circular, C unreachable-drug)
    def stratum(d, dis):
        if d in unreachable: return "C"
        return "B" if (d, dis) in circular else "A"
    pos = pos.with_columns(
        pl.struct(["drug", "disease"]).map_elements(
            lambda r: stratum(r["drug"], r["disease"]), return_dtype=pl.Utf8).alias("stratum"))

    s_neg, neg_keep, neg_drop = score_pairs(neg, model, e2i, rid, device)
    results = {}
    for label, pos_subset in (("full_AB", pos.filter(pl.col("stratum") != "C")),
                              ("A_only",  pos.filter(pl.col("stratum") == "A"))):
        s_pos, pos_keep, pos_drop = score_pairs(pos_subset, model, e2i, rid, device)
        if len(s_pos) == 0 or len(s_neg) == 0:
            results[label] = dict(n_pos=int(len(s_pos)), n_neg=int(len(s_neg)),
                                  AUROC=None, AUPRC=None, note="a class was empty after vocab filter")
            continue
        y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
        s = np.r_[s_pos, s_neg]
        auroc = round(float(roc_auc_score(y, s)), 4)
        auprc = round(float(average_precision_score(y, s)), 4)
        roc_ci, prc_ci = auc_ci(y, s)
        results[label] = dict(
            n_pos=int(len(s_pos)), n_neg=int(len(s_neg)),
            pos_dropped_vocab=int(pos_drop), neg_dropped_vocab=int(neg_drop),
            prevalence=round(len(s_pos) / (len(s_pos) + len(s_neg)), 4),   # AUPRC chance baseline
            AUROC=auroc, AUROC_95CI=roc_ci, AUPRC=auprc, AUPRC_95CI=prc_ci)

    summary = dict(split=split, ind_relation=IND,
                   n_pos_total=pos.height, n_neg_total=neg.height,
                   neg_dropped_vocab=int(neg_drop),
                   C_unreachable_pos=pos.filter(pl.col("stratum") == "C").height,
                   discrimination=results,
                   note="AUROC on held-out pos vs frozen-holdout neg; AUPRC baseline = prevalence. "
                        "full_AB = reachable positives; A_only = graph-independent (honest) positives.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["random", "disease_cold"],
                    choices=["random", "drug_cold", "disease_cold"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_reference()
    for split in args.splits:
        print(f"\n=== D5(b) discrimination: {split} ===")
        summary = evaluate_split(split, ref, device)
        json.dump(summary, open(OUT / f"d5b_discrimination_{split}.json", "w"), indent=2)
        for label, r in summary["discrimination"].items():
            if r.get("AUROC") is None:
                print(f"[{split}] {label}: {r['note']}"); continue
            print(f"[{split}] {label:8} pos={r['n_pos']} neg={r['n_neg']} prev={r['prevalence']}  "
                  f"AUROC={r['AUROC']} {r['AUROC_95CI']}  AUPRC={r['AUPRC']} {r['AUPRC_95CI']}")
    print(f"\nD5(b) complete -> {OUT}. LLM/agent columns append here post-B/C.")


if __name__ == "__main__":
    main()
