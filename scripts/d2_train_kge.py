#!/usr/bin/env python3
"""
ReTraceRx Phase D -- D2: Train the KGE baseline, one model per holdout split.

Consumes the factories D1 saved (artifacts/kge_data/<split>/{train_factory,test_factory}) and
produces one trained model per split (artifacts/kge_<model>_<split>/).

Reconciled with D1 / Phase A:
  * Loads the SAVED factories via from_path_binary -- never rebuilds (forward contract; a rebuild
    risks a different entity/relation map than the test_factory was aligned to).
  * One model per split (train indications differ per split: 693 / 5,707 / 5,493 excluded).
  * create_inverse_triples was set at D1 (D-D5); ranking drugs FOR a disease at D3 uses the inverse
    of the indication relation. Nothing to invert again here.
  * The pipeline's built-in MRR/Hits@k rank against ALL entities, unfiltered by the reachable
    whitelist and unstratified by circularity -> a TRAINING-HEALTH SMOKE TEST ONLY, not the D3 metric.

Terms (first use):
  KGE          Knowledge Graph Embedding -- a vector per node and per relation; a missing edge is
               scored by how compatible its endpoint vectors are under the relation.
  RotatE       a KGE model that represents each relation as a rotation in complex space; strong on
               biomedical graphs. TransE (translation) is the cheap sanity-check model.
  embedding_dim the length of each learned vector.
  epoch        one full pass over the training triples; batch_size = triples processed per step.
  sLCWA        stochastic Local Closed World Assumption -- the training regime that, for each true
               triple, samples corrupted (negative) triples the model must score lower.
  checkpoint   a periodic on-disk snapshot so an interrupted run resumes instead of restarting.

Run (headline first, then the optional third split):
  python scripts/d2_train_kge.py --splits random disease_cold --sanity
  python scripts/d2_train_kge.py --splits drug_cold
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

KGE_DATA = Path("artifacts/kge_data")
OUT      = Path("artifacts")


def load_factories(split: str):
    tr = TriplesFactory.from_path_binary(str(KGE_DATA / split / "train_factory"))
    te = TriplesFactory.from_path_binary(str(KGE_DATA / split / "test_factory"))
    inv = getattr(tr, "create_inverse_triples", None)
    print(f"[{split}] train_triples={tr.num_triples:,}  entities={tr.num_entities:,}  "
          f"relations={tr.num_relations}  inverse_triples={inv}  test_triples={te.num_triples}")
    assert inv, f"[{split}] train factory was built WITHOUT inverse triples -- rebuild D1 (D-D5)."
    return tr, te


def sanity_pass(split: str, tr, te, device: str):
    """A few epochs of TransE just to prove the loop runs, the GPU is used, and loss falls."""
    print(f"[{split}] sanity: TransE x3 epochs ...")
    r = pipeline(training=tr, testing=te, model="TransE", model_kwargs=dict(embedding_dim=64),
                 negative_sampler="basic", training_kwargs=dict(num_epochs=3, batch_size=4096),
                 random_seed=42, device=device)
    losses = r.losses
    ok = losses[-1] < losses[0]
    print(f"[{split}] sanity loss {losses[0]:.3f} -> {losses[-1]:.3f}  {'OK' if ok else 'NOT FALLING'}")
    return ok


def train_split(split: str, model: str, dim: int, epochs: int, batch: int, seed: int, device: str):
    tr, te = load_factories(split)

    # --- carve a SMALL validation factory from TRAIN (never from te / the D3 holdout) ---
    VAL_N = 5_000
    g = torch.Generator().manual_seed(seed)                 # deterministic carve
    perm = torch.randperm(tr.num_triples, generator=g)
    val_idx, core_idx = perm[:VAL_N], perm[VAL_N:]

    tr_core = tr.clone_and_exchange_triples(
        tr.mapped_triples[core_idx], create_inverse_triples=True)   # force the flag ON
    tr_val = tr.clone_and_exchange_triples(
        tr.mapped_triples[val_idx], create_inverse_triples=True)

    assert tr_core.create_inverse_triples and tr_val.create_inverse_triples
    assert tr_core.num_entities == tr.num_entities          # clone keeps tr's full id space
    print(f"[{split}] val carved from train: core={tr_core.num_triples:,}  val={tr_val.num_triples:,} "
          f"(te untouched: {te.num_triples})")

    out = OUT / f"kge_{model.lower()}_{split}"

    t0 = time.time()
    result = pipeline(
        training=tr_core, validation=tr_val, testing=te,
        model=model, model_kwargs=dict(embedding_dim=dim),
        negative_sampler="basic", negative_sampler_kwargs=dict(num_negs_per_pos=1),
        optimizer="Adam", optimizer_kwargs=dict(lr=1e-3),
        stopper="early",
        stopper_kwargs=dict(frequency=10, patience=3, relative_delta=0.002,
                            metric="inverse_harmonic_mean_rank"),
        training_kwargs=dict(num_epochs=epochs, batch_size=batch,
                             checkpoint_name=f"{model.lower()}_{split}.pt",
                             checkpoint_frequency=10),
        random_seed=seed, device=device,
    )

    dt = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(str(out))   # writes trained_model.pkl + training factory + results.json

    # built-in metrics are a SMOKE TEST ONLY (all-entity ranking) -- recorded, never reported as D3
    smoke = {}
    try:
        smoke = {"MRR_all_entities": result.get_metric("inverse_harmonic_mean_rank"),
                 "Hits@10_all_entities": result.get_metric("hits_at_10")}
    except Exception:
        pass
    meta = dict(split=split, model=model, embedding_dim=dim, epochs=epochs, batch_size=batch,
                seed=seed, device=device, wall_seconds=round(dt, 1),
                final_loss=(result.losses[-1] if result.losses else None),
                smoke_metrics_all_entity_ranking=smoke,
                note="smoke metrics rank against ALL entities; the project metric is D3 (whitelist+strata)")
    json.dump(meta, open(out / "d2_meta.json", "w"), indent=2)
    print(f"[{split}] saved -> {out}  final_loss={meta['final_loss']}  wall={dt/60:.1f} min  "
          f"(smoke {smoke})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["random", "disease_cold"],
                    choices=["random", "drug_cold", "disease_cold"])
    ap.add_argument("--model", default="RotatE", choices=["RotatE", "ComplEx", "TransE"])
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4096)   # L40S handles 8192+; lower if OOM
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanity", action="store_true", help="run a TransE smoke pass on split 1 first")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else "  (no GPU!)"))
    if device == "cpu":
        print("WARNING: training on CPU will be very slow; expected GPU on g6e.xlarge.")

    if args.sanity:
        tr, te = load_factories(args.splits[0])
        sanity_pass(args.splits[0], tr, te, device)

    for split in args.splits:
        train_split(split, args.model, args.embedding_dim, args.epochs,
                    args.batch_size, args.seed, device)

    print("\nD2 complete. Verify plumbing with:  python scripts/d2_verify_ranking.py --split disease_cold")
    print("Then proceed to D3 (stratified, whitelist-scoped evaluation).")


if __name__ == "__main__":
    main()