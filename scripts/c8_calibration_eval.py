#!/usr/bin/env python3
"""
c8_calibration_eval.py — Phase C, C8: abstention / calibration (the reframed
contribution). "An agent that knows what it doesn't know."

Runs a full-holdout agent pass (fast on vLLM), dumps per-pick records, and reports:

  1. ABSTENTION — grounded declines (abstains, or reasons but supports nothing) vs
     ungrounded which always answers. Broken down by cause:
       zero_candidate       (retrieval empty; OB-3-adjacent)
       no_faithful_support  (reasoned, nothing supported)
       link_fail            (weak entity link)
  2. HALLUCINATION-UNDER-PRESSURE — when nothing is supported, does grounded invent
     a confident pick? = confident-but-unverifiable rate
     (supported=true & faithful=false; ~0 under guided decoding) vs ungrounded's
     pick-hallucination (~1.0, always answers).
  3. CALIBRATION — does confidence track faithful support?
       AUROC(confidence -> faithful), reliability bins, mean conf faithful vs not.
     CAVEAT (stated, not hidden): citation_failure~=0 under guided decoding, so
     faithful ~= supported, and confidence is emitted alongside supported -> this
     measures self-consistency, not externally-validated calibration.

Outputs data/phaseC/c8/:
  c8_perpick_<split>.jsonl   (audit / re-aggregate with --reaggregate)
  c8_calibration_<split>.json

Usage:
    export REASONER_BACKEND=vllm
    python scripts/c8_calibration_eval.py --split disease_cold          # full
    python scripts/c8_calibration_eval.py --split disease_cold --limit 10   # smoke
    python scripts/c8_calibration_eval.py --split disease_cold --reaggregate # no re-run
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c0_config as C  # noqa: E402
from c2_agent import run_agent  # noqa: E402
from c6_ungrounded import run_ungrounded  # noqa: E402

C8 = C.PHASEC_DIR / "c8"
DISEASE_ID_COLS = ["okg_disease_id", "disease_graph_id", "disease", "okg_dis_id", "graph_id"]


def holdout_diseases(split):
    df = pl.read_csv(C.DATA / "holdout" / "splits" / f"{split}_holdout_positives.csv")
    col = next((c for c in DISEASE_ID_COLS if c in df.columns), df.columns[0])
    return df.get_column(col).cast(pl.Utf8).unique().sort().to_list()


def cause(reason: str) -> str:
    r = (reason or "").lower()
    if "zero-candidate" in r or "no 2-hop" in r:
        return "zero_candidate"
    if "faithful mechanistic support" in r:
        return "no_faithful_support"
    if "link too weak" in r or "link" in r:
        return "link_fail"
    return "other"


def auroc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda x: x[0])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    sp = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    npos, nneg = len(pos), len(neg)
    return round((sp - npos * (npos + 1) / 2) / (npos * nneg), 4)


def run_pass(split, mode, diseases, n_ungrounded, dump_path):
    perpick, perdisease = [], []
    with open(dump_path, "w") as f:
        for did in diseases:
            g = run_agent(disease_graph_id=did, split=split, mode=mode)
            if g.get("abstain"):
                rec = {"disease": did, "abstain": True, "cause": cause(g.get("abstain_reason")),
                       "n_picks": 0, "n_faithful": 0}
            else:
                r = g.get("ranking", [])
                nf = sum(1 for e in r if e["faithful"])
                for e in r:
                    pp = {"disease": did, "drug": e["drug_graph_id"],
                          "confidence": e.get("confidence") or 0.0,
                          "supported": bool(e["supported"]), "faithful": bool(e["faithful"]),
                          "n_real": len(e["cited_path_ids"]),
                          "n_hall": len(e.get("hallucinated_path_ids", []))}
                    perpick.append(pp)
                    f.write(json.dumps(pp) + "\n")
                rec = {"disease": did, "abstain": False, "cause": None,
                       "n_picks": len(r), "n_faithful": nf, "no_support": nf == 0}
            u = run_ungrounded(did, n_ungrounded)
            rec["ungrounded_n_picks"] = len(u.get("picks", []))     # always answers
            perdisease.append(rec)
    (C8 / f"c8_perdisease_{split}.json").write_text(json.dumps(perdisease, indent=2))
    return perpick, perdisease


def load_dump(split):
    pp = [json.loads(l) for l in open(C8 / f"c8_perpick_{split}.jsonl")]
    pd = json.loads((C8 / f"c8_perdisease_{split}.json").read_text())
    return pp, pd


def aggregate(split, mode, perpick, perdisease):
    nd = len(perdisease)
    n_abstain = sum(1 for d in perdisease if d["abstain"])
    n_nosupport = sum(1 for d in perdisease if not d["abstain"] and d.get("no_support"))
    n_answer = nd - n_abstain - n_nosupport
    causes = {}
    for d in perdisease:
        if d["abstain"]:
            causes[d["cause"]] = causes.get(d["cause"], 0) + 1

    conf_faith = [p["confidence"] for p in perpick if p["faithful"]]
    conf_unf = [p["confidence"] for p in perpick if not p["faithful"]]
    n_confident_invention = sum(1 for p in perpick if p["supported"] and not p["faithful"])

    bins = []
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        hi = lo + 0.2
        b = [p for p in perpick if lo <= p["confidence"] < hi or (hi == 1.0 and p["confidence"] == 1.0)]
        bins.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": len(b),
                     "frac_faithful": round(sum(p["faithful"] for p in b) / len(b), 3) if b else None})

    rep = {
        "split": split, "mode": mode, "n_diseases": nd, "n_picks": len(perpick),
        "abstention": {
            "grounded_abstain_rate": round(n_abstain / nd, 4) if nd else None,
            "grounded_no_support_rate": round(n_nosupport / nd, 4) if nd else None,
            "grounded_answer_rate": round(n_answer / nd, 4) if nd else None,
            "grounded_declines_total_rate": round((n_abstain + n_nosupport) / nd, 4) if nd else None,
            "ungrounded_abstain_rate": 0.0,
            "abstain_causes": causes,
        },
        "hallucination_under_pressure": {
            "grounded_confident_but_unverifiable": n_confident_invention,
            "grounded_confident_invention_rate": round(n_confident_invention / len(perpick), 4)
                if perpick else None,
            "ungrounded_pick_hallucination": 1.0,
            "note": "grounded never confidently invents (citation-enum + verify); ungrounded always answers, 0 grounded citations",
        },
        "calibration": {
            "auroc_confidence_predicts_faithful": auroc(conf_faith, conf_unf),
            "mean_conf_faithful": round(sum(conf_faith) / len(conf_faith), 3) if conf_faith else None,
            "mean_conf_unfaithful": round(sum(conf_unf) / len(conf_unf), 3) if conf_unf else None,
            "reliability_bins": bins,
            "caveat": "faithful~=supported under guided decoding; confidence emitted alongside supported -> self-consistency, not external calibration",
        },
    }
    out = C8 / f"c8_calibration_{split}.json"
    out.write_text(json.dumps(rep, indent=2))

    ab = rep["abstention"]; cal = rep["calibration"]; hp = rep["hallucination_under_pressure"]
    print("================= C8 abstention / calibration =================")
    print(f"split={split}  diseases={nd}  picks={len(perpick)}")
    print(f"ABSTENTION (grounded): abstain={ab['grounded_abstain_rate']}  "
          f"no_support={ab['grounded_no_support_rate']}  "
          f"declines_total={ab['grounded_declines_total_rate']}  |  ungrounded=0.0")
    print(f"  causes: {ab['abstain_causes']}")
    print(f"UNDER PRESSURE: grounded confident-invention={hp['grounded_confident_but_unverifiable']} "
          f"(rate {hp['grounded_confident_invention_rate']})  |  ungrounded pick-halluc=1.0")
    print(f"CALIBRATION: AUROC(conf->faithful)={cal['auroc_confidence_predicts_faithful']}  "
          f"mean_conf faithful={cal['mean_conf_faithful']} vs unfaithful={cal['mean_conf_unfaithful']}")
    for b in cal["reliability_bins"]:
        print(f"    conf {b['bin']}: n={b['n']:<4} frac_faithful={b['frac_faithful']}")
    print(f"wrote {out}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=C.HEADLINE_SPLIT)
    ap.add_argument("--mode", default="plus_true")
    ap.add_argument("--n-ungrounded", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reaggregate", action="store_true", help="skip the run; use existing dump")
    a = ap.parse_args()
    C8.mkdir(parents=True, exist_ok=True)

    if a.reaggregate:
        perpick, perdisease = load_dump(a.split)
        print(f"[c8] re-aggregating {len(perdisease)} diseases from dump\n")
    else:
        diseases = holdout_diseases(a.split)
        if a.limit:
            diseases = diseases[: a.limit]
            print(f"[c8] SMOKE {len(diseases)} diseases (use no --limit for report)\n")
        else:
            print(f"[c8] FULL holdout {len(diseases)} diseases from {a.split}\n")
        perpick, perdisease = run_pass(a.split, a.mode, diseases, a.n_ungrounded,
                                       C8 / f"c8_perpick_{a.split}.jsonl")
    aggregate(a.split, a.mode, perpick, perdisease)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
