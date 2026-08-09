#!/usr/bin/env python3
"""
e6_pick_demos.py — rank holdout diseases by how strong a live demo they make.

For each disease in a split's holdout, probe the RUNNING e2_api the same way the
UI does, and score it on what a viewer would actually see:
  * grounded recovers the held-out true drug (not abstained), at a low rank
  * that recovered pick is verified (faithful + supported, citations resolve)
  * the true drug has a recognizable name (from the API's drug_name enrichment)
  * BONUS: the ungrounded baseline MISSES it — the ideal grounding contrast

Writes data/phaseE/e6_demo_candidates_<split>_<mode>.csv and prints the top rows.
This is a curation aid, not an eval — recovery here is a plain id match against
the grounded ranking, using the live stack (so it matches the demo exactly).

Run (serve_env sourced, e2_api up on :8080):
    python scripts/e6_pick_demos.py                       # disease_cold, plus_true
    python scripts/e6_pick_demos.py --split random --mode whitelist
    python scripts/e6_pick_demos.py --max 8 --no-ungrounded    # faster

Modes: plus_true = capability framing (true drug is in the universe, so recovery
measures ranking) — best for finding diseases that demo the agent's strength.
whitelist = deployment framing (true drug only if reachable); use to check what
the UI's default shows.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_config as C


def _pick_col(cols, keys):
    low = {c.lower(): c for c in cols}
    for k in keys:
        for lc, orig in low.items():
            if k in lc:
                return orig
    return None


def load_positives(path: Path):
    """→ ({disease_id: set(true_drug_ids)}, detected_cols). Auto-detects the
    disease-id and drug-id columns and prints what it found."""
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return {}, {}
    cols = list(rows[0].keys())
    dcol = _pick_col(cols, ["disease_graph_id", "disease_id", "disease", "dis_id", "mondo"])
    kcol = _pick_col(cols, ["drug_graph_id", "drug_id", "true_drug", "chembl", "drug"])
    if not dcol or not kcol:
        raise SystemExit(f"could not detect disease/drug columns in {cols}; "
                         f"pass --disease-col / --drug-col")
    pos: dict[str, set] = {}
    for r in rows:
        d, k = r.get(dcol), r.get(kcol)
        if d and k:
            pos.setdefault(d, set()).add(k)
    return pos, {"disease_col": dcol, "drug_col": kcol}


def probe(base, gid, split, mode, want_ungrounded):
    """Return a per-disease record from the live API."""
    rec = {"disease_graph_id": gid, "abstain": False, "grounded_rank": None,
           "drug_name": "", "verified": False, "confidence": None,
           "ungrounded_hit": None, "n_ranked": 0}
    g = httpx.post(f"{base}/repurpose", timeout=300, json={
        "disease_graph_id": gid, "split": split, "mode": mode, "grounded": True}).json()
    if g.get("needs_disambiguation"):
        rec["abstain"] = True; rec["note"] = "did not link"; return rec, set()
    if g.get("abstain"):
        rec["abstain"] = True; return rec, set()
    ranking = g.get("ranking", [])
    rec["n_ranked"] = len(ranking)
    return rec, ranking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=C.HEADLINE_SPLIT, choices=C.SPLITS)
    ap.add_argument("--mode", default="plus_true", choices=["plus_true", "whitelist"])
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--top-k", type=int, default=10, help="recovery counts if rank ≤ this")
    ap.add_argument("--max", type=int, default=None, help="cap diseases probed")
    ap.add_argument("--positives", default=None, help="override holdout positives CSV")
    ap.add_argument("--no-ungrounded", dest="ungrounded", action="store_false",
                    help="skip the ungrounded contrast probe (faster)")
    args = ap.parse_args()

    pos_path = Path(args.positives) if args.positives else C.split_positives(args.split)
    if not pos_path.exists():
        raise SystemExit(f"holdout positives not found: {pos_path}")
    positives, detected = load_positives(pos_path)
    print(f"[e6] {pos_path.name}: {len(positives)} diseases  (cols: {detected})")

    diseases = list(positives)[: args.max] if args.max else list(positives)
    rows = []
    for i, gid in enumerate(diseases, 1):
        true_set = positives[gid]
        print(f"  [{i}/{len(diseases)}] {gid} …", end=" ", flush=True)
        try:
            rec, ranking = probe(args.base, gid, args.split, args.mode, args.ungrounded)
        except Exception as e:  # noqa: BLE001
            print(f"ERR {type(e).__name__}"); rec = {"disease_graph_id": gid, "error": str(e)}
            rows.append(rec); continue

        # recovery: first ranked pick whose drug id is a held-out true drug
        for rank, cand in enumerate(ranking, 1):
            if cand.get("drug_graph_id") in true_set:
                rec["grounded_rank"] = rank
                rec["drug_name"] = cand.get("drug_name") or cand.get("drug_graph_id")
                rec["verified"] = bool(cand.get("faithful") and cand.get("supported")
                                       and not cand.get("hallucinated_path_ids"))
                rec["confidence"] = cand.get("confidence")
                break

        # ungrounded contrast: did the baseline surface any true drug?
        if args.ungrounded and not rec.get("abstain"):
            try:
                u = httpx.post(f"{args.base}/repurpose", timeout=300, json={
                    "disease_graph_id": gid, "split": args.split,
                    "grounded": False}).json()
                uids = set(u.get("ranked_drug_ids", []) or
                           [p.get("drug_graph_id") for p in u.get("picks", [])])
                rec["ungrounded_hit"] = bool(true_set & uids)
            except Exception:  # noqa: BLE001
                rec["ungrounded_hit"] = None

        rec["recognizable"] = bool(rec.get("drug_name") and
                                   rec["drug_name"] != rec.get("disease_graph_id") and
                                   any(c.isalpha() for c in rec.get("drug_name", "")))
        rec["demo_score"] = _score(rec, args.top_k)
        print(f"rank={rec['grounded_rank']} verified={rec['verified']} "
              f"ung_hit={rec['ungrounded_hit']} score={rec['demo_score']}")
        rows.append(rec)

    rows.sort(key=lambda r: r.get("demo_score", -1), reverse=True)

    out = C.PHASEE_OUT / f"e6_demo_candidates_{args.split}_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["disease_graph_id", "drug_name", "grounded_rank", "verified", "confidence",
            "ungrounded_hit", "recognizable", "demo_score", "n_ranked", "abstain", "note", "error"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n=== best demo candidates (grounded recovers a recognizable true drug, "
          "ideally where ungrounded misses) ===")
    shown = [r for r in rows if r.get("grounded_rank")][:10]
    for r in shown:
        contrast = "  ← ungrounded MISSES" if r.get("ungrounded_hit") is False else ""
        print(f"  score {r['demo_score']:>3}  rank {r['grounded_rank']:<2}  "
              f"{'✓' if r['verified'] else '·'}  {r.get('drug_name','')}"
              f"  ({r['disease_graph_id']}){contrast}")
    if not shown:
        print("  (none recovered a true drug in the pool — expected under the 2-hop "
              "ceiling; these are the honest-abstention diseases)")
    print(f"\nfull table → {out}")


def _score(r, top_k):
    s = 0
    gr = r.get("grounded_rank")
    if gr and not r.get("abstain"):
        s += max(0, 100 - gr * 3)                 # recovery, reward low rank
        if gr <= top_k:
            s += 10
    if r.get("verified"):
        s += 20
    if r.get("recognizable"):
        s += 15
    if gr and r.get("ungrounded_hit") is False:   # the money-shot contrast
        s += 40
    return s


if __name__ == "__main__":
    main()
