#!/usr/bin/env python3
"""
ReTraceRx Phase D -- D4: The shared evaluation harness (one scorer, many rankers).

ONE metric engine, MANY pluggable rankers. Every ranker is scored on the identical holdout, reachable
whitelist, circularity strata, filtering, and ceilings -- so the ablation cannot be voided by a scoring
divergence between systems (how-to D4 "watch out"). This generalizes D3: running the KGE ranker through
here reproduces the D3 numbers exactly (use that as the refactor correctness check).

Ranker interface (all systems implement this, now and later):
    rank(disease: str, universe: set[str]) -> list[str]   # drug ids best-first, restricted to `universe`

Rankers:
  * KGERanker         -- READY NOW (D2 model + D3 ranking).
  * MockRanker        -- deterministic shuffle, to test the harness with no model.
  * UngroundedLLMRanker -- POST-PHASE-B: needs the name->CURIE linker. Interface fixed; body deferred.
  * GroundedAgentRanker -- POST-PHASE-B+C: needs retrieval + agent + verifier. Interface fixed; deferred.

Reuses D3's tested reference/metric builders (no metric-logic duplication -> no drift).

Run:
  python scripts/d4_harness.py --ranker kge  --splits random disease_cold drug_cold
  python scripts/d4_harness.py --ranker mock --splits random          # harness smoke test
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Protocol
import polars as pl
import torch
from pykeen.triples import TriplesFactory
try:
    from pykeen.predict import predict_target
    _NEW = True
except Exception:
    _NEW = False

# reuse D3's reference + metric builders verbatim (single source of truth)
from d3_evaluate_kge import (load_reference, metrics, apply_remap,
                             KGE_DATA, SPLIT_POS)

OUT = Path("artifacts/d4_harness")


# --------------------------------------------------------------- ranker interface + implementations
class Ranker(Protocol):
    name: str
    def rank(self, disease: str, universe: set[str]) -> list[str]: ...


class KGERanker:
    """READY NOW. Wraps the D2 RotatE model; identical scoring to D3."""
    name = "kge_rotate"
    def __init__(self, split: str, device: str):
        self.tf = TriplesFactory.from_path_binary(str(KGE_DATA / split / "train_factory"))
        self.IND = json.load(open(KGE_DATA / split / "d1_meta.json"))["ind_relation"]
        assert self.IND in self.tf.relation_to_id, f"{self.IND!r} not in model relation vocab"
        self.model = torch.load(f"artifacts/kge_rotate_{split}/trained_model.pkl",
                                weights_only=False).to(device)
        self.model.eval()
    def rank(self, disease, universe):
        if _NEW:
            df = predict_target(model=self.model, tail=disease, relation=self.IND,
                                triples_factory=self.tf).df
            col = "head_label" if "head_label" in df.columns else "head_id"
        else:
            df = self.model.get_head_prediction_df(relation_label=self.IND, tail_label=disease,
                                                   triples_factory=self.tf)
            col = "head_label"
        df = df[df[col].isin(universe)].sort_values("score", ascending=False)
        return df[col].tolist()


class MockRanker:
    """Deterministic shuffle -- lets you exercise the whole harness with no trained model."""
    name = "mock"
    def __init__(self, split: str, device: str, seed: int = 0):
        self.rng = random.Random(seed)
    def rank(self, disease, universe):
        u = list(universe); self.rng.shuffle(u); return u


class UngroundedLLMRanker:
    """POST-PHASE-B. Prompt Qwen 3.5-9B (closed-book) for a ranked drug list, map names->CURIEs via
    the Phase B linker, intersect with `universe`. Interface is frozen now; wire when the linker exists."""
    name = "ungrounded_llm"
    def __init__(self, *a, **k):
        raise NotImplementedError("UngroundedLLMRanker needs the Phase B name->CURIE linker (post-Phase-B).")
    def rank(self, disease, universe):  # signature fixed so the harness never changes
        raise NotImplementedError


class GroundedAgentRanker:
    """POST-PHASE-B+C. Retrieval + reasoning + faithfulness; returns ranked candidates with cited paths.
    The how-to marks this '(after Phase C)'. Interface frozen now; wire when the agent exists."""
    name = "grounded_agent"
    def __init__(self, *a, **k):
        raise NotImplementedError("GroundedAgentRanker needs Phase B retrieval + Phase C agent (post-B+C).")
    def rank(self, disease, universe):
        raise NotImplementedError


RANKERS = {"kge": KGERanker, "mock": MockRanker,
           "ungrounded_llm": UngroundedLLMRanker, "grounded_agent": GroundedAgentRanker}


# --------------------------------------------------------------- the shared metric engine
def evaluate(ranker, split, candidate_mode, ref):
    reachable, unreachable, circular, known0, remap, reach_counts = ref
    known = {k: set(v) for k, v in known0.items()}          # per-split copy -> no cross-split leak (ED9)

    pos = (pl.read_csv(SPLIT_POS(split)).select(drug="okg_drug_id", disease="okg_disease_id")
             .pipe(apply_remap, remap, "disease").unique())
    split_true = set(pos["drug"].to_list())
    for d, dis in zip(pos["drug"].to_list(), pos["disease"].to_list()):
        known.setdefault(dis, set()).add(d)
    universe = reachable | split_true                       # fixed pool the ranker orders per disease

    by_dis: dict[str, list] = {}
    for d, dis in zip(pos["drug"].to_list(), pos["disease"].to_list()):
        by_dis.setdefault(dis, []).append(d)

    order_cache: dict[str, list] = {}
    records = []
    for dis, drugs in by_dis.items():
        for true_drug in drugs:
            if true_drug in unreachable:
                records.append((ranker.name, split, candidate_mode, dis, true_drug, "C_unreachable", None)); continue
            if candidate_mode == "whitelist" and true_drug not in reachable:
                records.append((ranker.name, split, candidate_mode, dis, true_drug, "C_not_candidate", None)); continue
            stratum = "B_circular" if (true_drug, dis) in circular else "A_independent"
            if dis not in order_cache:
                order_cache[dis] = ranker.rank(dis, universe)      # ranker sees ONE universe per disease
            ordering = order_cache[dis]
            pool = (reachable | {true_drug}) if candidate_mode == "plus_true" else reachable
            ranked = [d for d in ordering if d in pool]
            others = known.get(dis, set()) - {true_drug}
            filtered = [d for d in ranked if d == true_drug or d not in others]
            rank = (filtered.index(true_drug) + 1) if true_drug in filtered else None
            records.append((ranker.name, split, candidate_mode, dis, true_drug, stratum, rank))

    rec = pl.DataFrame(records,
                       schema=["ranker", "split", "candidate_mode", "disease", "drug", "stratum", "rank"],
                       orient="row")

    def r(s): return rec.filter(pl.col("stratum") == s)["rank"].to_list()
    A, B = r("A_independent"), r("B_circular")
    C_un_pairs = rec.filter(pl.col("stratum") == "C_unreachable").height
    C_un_drugs = rec.filter(pl.col("stratum") == "C_unreachable")["drug"].n_unique()
    C_nc = rec.filter(pl.col("stratum") == "C_not_candidate").height
    n = rec.height; a8 = reach_counts.get(split)
    summary = dict(
        ranker=ranker.name, split=split, candidate_mode=candidate_mode, n_positives=n,
        stratum_A_independent=metrics(A), stratum_B_circular=metrics(B),
        reachable_scored_AB=metrics(A + B),
        full_with_ceiling_as_miss=metrics(A + B + [None] * (C_un_pairs + C_nc)),
        ceiling=dict(C_unreachable_pairs=C_un_pairs, C_unreachable_drugs=C_un_drugs,
                     C_not_candidate_pairs=C_nc, a8_unreachable_drug_count=a8,
                     checksum_ok=(a8 is None or C_un_drugs == a8),
                     max_achievable_recall=round((n - C_un_pairs - C_nc) / n, 4) if n else None),
    )
    return summary, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranker", required=True, choices=list(RANKERS))
    ap.add_argument("--splits", nargs="+", default=["random", "disease_cold"],
                    choices=["random", "drug_cold", "disease_cold"])
    ap.add_argument("--candidate-mode", choices=["plus_true", "whitelist"], default="plus_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    ref = load_reference()
    all_rec = []
    for split in args.splits:
        ranker = RANKERS[args.ranker](split, device)       # raises for deferred rankers, by design
        print(f"\n=== D4 harness: ranker={ranker.name}  split={split}  mode={args.candidate_mode} ===")
        summary, rec = evaluate(ranker, split, args.candidate_mode, ref)
        json.dump(summary, open(OUT / f"d4_{ranker.name}_{split}_{args.candidate_mode}.json", "w"), indent=2)
        all_rec.append(rec)
        a, c = summary["stratum_A_independent"], summary["ceiling"]
        print(f"[{split}] A n={a['n']} Hits@10={a['Hits@10']} MRR={a['MRR']}  "
              f"B n={summary['stratum_B_circular']['n']}  "
              f"C={c['C_unreachable_pairs']}p/{c['C_unreachable_drugs']}d (ok={c['checksum_ok']})  "
              f"max_recall={c['max_achievable_recall']}")
    # one long-format per-query table across splits -> the D6/D7/D8 substrate (all systems append here)
    pl.concat(all_rec).write_csv(OUT / f"d4_perquery_{args.ranker}_{args.candidate_mode}.csv")
    print(f"\nD4 complete -> {OUT}. When Phase B/C land, their rankers append rows in this same schema.")


if __name__ == "__main__":
    main()
