#!/usr/bin/env python3
"""
Diagnose why d1 read `a8 unreachable = 0` when the record expects 23/16/20.

It maps a8_reachability.json without assuming any shape, then reports every JSON
path *(the sequence of keys/indices from the root to a value)* that plausibly holds
the per-split unreachable positives, and shows what a handful of candidate accessors
actually return. No data files are modified.

Run:  python scripts/check_a8_reachability.py
"""
import json, re
from pathlib import Path

REACH = Path("data/manifest/a8_reachability.json")     # match your d1 COLLAPSE_PQ fix location;
if not REACH.exists():                                 # fall back to the holdout dir if needed
    alt = Path("data/holdout/a8_reachability.json")
    REACH = alt if alt.exists() else REACH

SPLITS   = ["random", "drug_cold", "disease_cold"]
EXPECTED = {"random": 23, "drug_cold": 16, "disease_cold": 20}   # from the status record
KEY_HINT = re.compile(r"reach|unreach|positive|holdout|split|miss|ceiling", re.I)


def walk(node, path=()):
    """Yield (path_tuple, value) for every node in the JSON tree."""
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, path + (i,))


def fmt(path):
    return "".join(f"[{k!r}]" if isinstance(k, str) else f"[{k}]" for k in path)


def summarize(v):
    if isinstance(v, dict):
        return f"dict(keys={list(v)[:8]}{'...' if len(v) > 8 else ''})"
    if isinstance(v, list):
        head = v[:3]
        return f"list(len={len(v)}, head={head})"
    return f"{type(v).__name__}({v!r})"


def main():
    print(f"Reading {REACH}\n")
    data = json.load(open(REACH))

    # 1) top-level shape
    print("=== top-level keys ===")
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"  {k!r:32} -> {summarize(v)}")
    else:
        print(f"  root is a {type(data).__name__}: {summarize(data)}")

    # 2) every path whose KEY name hints at reachability, plus what it holds
    print("\n=== paths with reachability-like key names ===")
    for path, v in walk(data):
        if path and isinstance(path[-1], str) and KEY_HINT.search(path[-1]):
            print(f"  {fmt(path):50} {summarize(v)}")

    # 3) any leaf list whose length equals an EXPECTED split count (23/16/20)
    print("\n=== lists whose length matches an expected count (23/16/20) ===")
    exp_vals = set(EXPECTED.values())
    for path, v in walk(data):
        if isinstance(v, list) and len(v) in exp_vals:
            print(f"  len={len(v):<4} {fmt(path)}   head={v[:3]}")

    # 4) any dict that looks split-keyed (keys are a superset of the split names)
    print("\n=== dicts keyed by split name ===")
    for path, v in walk(data):
        if isinstance(v, dict) and any(s in v for s in SPLITS):
            present = {s: (len(v[s]) if isinstance(v.get(s), list) else v.get(s)) for s in SPLITS if s in v}
            print(f"  {fmt(path):50} split->count {present}")

    # 5) show what d1's current accessor returns vs. the best candidate
    print("\n=== accessor check ===")
    def count_at(container, split):
        try:
            val = container[split]
            return len(val) if isinstance(val, list) else val
        except Exception as e:
            return f"<{type(e).__name__}>"

    current = data.get("unreachable_positives") if isinstance(data, dict) else None
    print(f"  d1 currently uses: data['unreachable_positives'][<split>]")
    for s in SPLITS:
        got = count_at(current, s) if isinstance(current, (dict, list)) else "<key 'unreachable_positives' absent>"
        flag = "" if got == EXPECTED[s] else f"   != expected {EXPECTED[s]}"
        print(f"    {s:14} -> {got}{flag}")

    print("\nPick the path from sections 3/4 whose split counts read 23/16/20,")
    print("then set d1's accessor to that path (e.g. reach[<newkey>][split]).")


if __name__ == "__main__":
    main()
