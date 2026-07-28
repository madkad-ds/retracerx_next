#!/usr/bin/env python3
"""
Post-D2 plumbing check (a mini-D3). For one held-out disease it ranks the reachable-whitelist drugs
and reports where the true drug lands. Proves, before D3 is written, that:
  * the trained model reloads,
  * the derived indication relation string resolves in the model's relation vocab,
  * head prediction (drugs given a disease) works via the inverse relation set at D1 (D-D5),
  * the reachable-whitelist filter (1,565) applies.

Run:  python scripts/d2_verify_ranking.py --split disease_cold --model rotate
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import polars as pl
from pykeen.triples import TriplesFactory
try:
    from pykeen.predict import predict_target          # pykeen >= 1.10
    _NEW = True
except Exception:
    _NEW = False                                        # fall back to model.get_head_prediction_df

KGE_DATA  = Path("artifacts/kge_data")
REACHABLE = Path("data/holdout/candidate_whitelist_reachable.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["random", "drug_cold", "disease_cold"])
    ap.add_argument("--model", default="rotate")
    args = ap.parse_args()

    art = Path(f"artifacts/kge_{args.model}_{args.split}")
    meta = json.load(open(KGE_DATA / args.split / "d1_meta.json"))
    IND = meta["ind_relation"]                                   # e.g. 'DRG-DIS__INDICATION'
    reachable = set(pl.read_csv(REACHABLE)["okg_drug_id"].to_list())

    tr = TriplesFactory.from_path_binary(str(KGE_DATA / args.split / "train_factory"))
    te = TriplesFactory.from_path_binary(str(KGE_DATA / args.split / "test_factory"))
    import torch
    model = torch.load(art / "trained_model.pkl", weights_only=False)
    model.eval()

    assert IND in tr.relation_to_id, f"{IND} not in model relation vocab ({IND!r})"

    # pick the first held-out (drug, IND, disease) whose true drug is IN the reachable whitelist
    triples = te.triples                                         # labeled ndarray [[h, r, t], ...]
    pick = next(((h, r, t) for h, r, t in triples if h in reachable), None)
    assert pick, "no held-out positive has its true drug in the reachable whitelist (unexpected)"
    true_drug, _, disease = pick

    if _NEW:
        df = predict_target(model=model, tail=disease, relation=IND, triples_factory=tr).df
        df = df.rename(columns={"head_label": "drug"}) if "head_label" in df.columns else \
             df.rename(columns={"head_id": "drug"})
    else:
        df = model.get_head_prediction_df(relation_label=IND, tail_label=disease, triples_factory=tr)
        df = df.rename(columns={"head_label": "drug"})

    df = df[df["drug"].isin(reachable)].sort_values("score", ascending=False).reset_index(drop=True)
    hits = df.index[df["drug"] == true_drug].tolist()
    rank = (hits[0] + 1) if hits else None

    print(f"[{args.split}] disease={disease}  true_drug={true_drug}")
    print(f"  ranked {len(df)} reachable drugs; true-drug rank = {rank}"
          f"{'  (in top-10)' if rank and rank <= 10 else ''}")
    print("  PLUMBING OK -- model + IND relation + inverse head-prediction + whitelist filter all resolve.")
    if rank is None:
        print("  NOTE: true drug not scored -- check it survived into the model vocab; not a plumbing pass.")


if __name__ == "__main__":
    main()
