#!/usr/bin/env python3
"""
ReTraceRx Phase B — B1: build the linkable DISEASE dictionary. (v2 — scoped surfaces)

Emits (surface_form -> node) rows so B2 can map a free-text disease mention to a
real OptimusKG node. Corrected to the ACTUAL Phase A schema
(RETRACERX_PHASE_A_STATUS_COMPLETED.md wins over the Phase B how-to):

  * source path : data/canon/canon_nodes.parquet   (NOT data/canon_nodes.parquet)
  * label column: `label`, disease value "DIS"      (NOT category == "Disease")
  * id column   : `id` (verbatim; decision D4)       (NOT id_curie)
  * synonyms    : inside `properties_json` (a verbatim JSON string)
  * OBA nodes   : dropped (erratum E5; matches A6 disease_xwalk's OBA exclusion)

SURFACE SCOPING (v2). The release exposes OBO synonym SCOPES separately
(exact/broad/narrow/related). They are NOT interchangeable for entity linking:
  * exact_synonyms  -> concept-equivalent           -> KEPT  (src "synonym:exact")
  * broad_synonyms  -> BROADER than the node         -> DROPPED (kills precision)
  * narrow_synonyms -> narrower than the node        -> DROPPED
  * related_synonyms-> loosely related               -> DROPPED
Plus high-quality clinical surfaces for recall (toggle INCLUDE_RECALL_SURFACES):
  * concept_names, snomed_full_names -> KEPT (src "concept_name"/"snomed_full_name")
`src` records the scope so B2 can weight matches into link_score instead of
treating every surface as equal.

Two identifiers per row, NOT the same string:
  * graph_id : verbatim `id` Neo4j stores and B3's Cypher matches on ("MONDO_0050890")
  * curie    : colon-form display CURIE for the contract JSON       ("MONDO:0050890")

Output:
  data/retrieval/link_dict_disease.parquet
  data/retrieval/b1_link_dict_stats.json
"""

from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path

import polars as pl

SRC = Path("data/canon/canon_nodes.parquet")
OUT_DIR = Path("data/retrieval")
OUT_PARQUET = OUT_DIR / "link_dict_disease.parquet"
OUT_STATS = OUT_DIR / "b1_link_dict_stats.json"

DISEASE_LABEL = "DIS"
OBA_PREFIX = "OBA"
STRICT_ATTR_DROP = False
ATTR_PREFIXES = ("PATO", "MP", "OBI", "OGMS")  # negligible counts (<=4 each)

# --- surface-form sources: precision-first allow-list (properties_json key -> src tag) ---
SURFACE_KEYS = {"exact_synonyms": "synonym:exact"}
RECALL_KEYS = {"concept_names": "concept_name", "snomed_full_names": "snomed_full_name"}
INCLUDE_RECALL_SURFACES = True
# scopes deliberately excluded — reported (count) but never emitted as surfaces
EXCLUDED_SCOPE_KEYS = ("broad_synonyms", "narrow_synonyms", "related_synonyms")

_WS = re.compile(r"\s+")


def norm_surface(s: str) -> str:
    """Normalize a surface form for matching: NFKC, strip, lowercase, collapse ws."""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return _WS.sub(" ", s)


def flatten_strings(obj, out: list, depth: int = 0):
    """Collect non-empty string leaves from a value (list / {'val':..} object / str)."""
    if depth > 6:
        return
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            flatten_strings(x, out, depth + 1)
    elif isinstance(obj, dict):
        for v in obj.values():
            flatten_strings(v, out, depth + 1)


def to_display_curie(id_str: str, id_prefix, norm_val) -> str:
    """Colon-form CURIE for the contract. Prefer a derived norm column; else derive
    from `id` (DOID_0050890 -> DOID:0050890) only when the id is prefixed."""
    if norm_val:
        return norm_val
    if id_prefix and "_" in id_str:
        return id_str.replace("_", ":", 1)
    return id_str  # delimiter-free ids (E19) stay verbatim


def main():
    if not SRC.exists():
        raise FileNotFoundError(
            f"{SRC} not found. Phase A writes canon_nodes under data/canon/ "
            "(the Phase B how-to's data/canon_nodes.parquet path is wrong)."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    active_keys = dict(SURFACE_KEYS)
    if INCLUDE_RECALL_SURFACES:
        active_keys.update(RECALL_KEYS)

    nodes = pl.read_parquet(SRC)
    if "properties_json" not in nodes.columns:
        print("WARN: no properties_json column — dictionary will be name-only.")
    norm_col = next(
        (c for c in ("id_norm", "norm", "id_curie_norm") if c in nodes.columns), None
    )

    dis = nodes.filter(pl.col("label") == DISEASE_LABEL)
    n_dis_all = dis.height
    dis = dis.filter(~pl.col("id").str.starts_with(OBA_PREFIX))  # E5 / A6 parity
    if STRICT_ATTR_DROP:
        for p in ATTR_PREFIXES:
            dis = dis.filter(~pl.col("id").str.starts_with(p))
    n_dis_kept = dis.height

    prefix_dist = (
        dis.with_columns(
            pl.col("id").str.splitn("_", 2).struct.field("field_0").alias("prefix")
        )
        .group_by("prefix")
        .len()
        .sort("len", descending=True)
    )

    cols = ["id", "label", "name", "id_prefix"]
    if norm_col:
        cols.append(norm_col)
    if "properties_json" in dis.columns:
        cols.append("properties_json")
    sub = dis.select([c for c in cols if c in dis.columns])

    # --- report which surface keys are present / excluded (transparency, E6) ---
    seen_top_keys, excluded_present = set(), set()
    if "properties_json" in sub.columns:
        for r in sub.head(500).iter_rows(named=True):
            blob = r.get("properties_json")
            if not blob:
                continue
            try:
                d = json.loads(blob)
            except Exception:
                continue
            if isinstance(d, dict):
                for k in d.keys():
                    seen_top_keys.add(str(k))
                    if str(k) in EXCLUDED_SCOPE_KEYS:
                        excluded_present.add(str(k))
    print("kept surface keys:", list(active_keys.keys()) + ["name"])
    print("excluded scope keys present in sample:", sorted(excluded_present) or "NONE")

    # --- build surface rows ---
    records = []
    n_blob_err = 0
    excluded_skipped = 0  # count of surfaces we deliberately did NOT emit
    for r in sub.iter_rows(named=True):
        gid = r["id"]
        disp = to_display_curie(gid, r.get("id_prefix"), r.get(norm_col) if norm_col else None)

        surfaces: list[tuple[str, str]] = []
        nm = r.get("name")
        if nm and str(nm).strip():
            surfaces.append((str(nm), "name"))

        blob = r.get("properties_json")
        if blob:
            try:
                parsed = json.loads(blob)
                if isinstance(parsed, dict):
                    for key, tag in active_keys.items():
                        if key in parsed:
                            vals: list[str] = []
                            flatten_strings(parsed[key], vals)
                            for s in vals:
                                surfaces.append((s, tag))
                    for key in EXCLUDED_SCOPE_KEYS:
                        if key in parsed:
                            skip: list[str] = []
                            flatten_strings(parsed[key], skip)
                            excluded_skipped += len(skip)
            except Exception:
                n_blob_err += 1

        for raw, src in surfaces:
            sn = norm_surface(raw)
            if not sn:
                continue
            records.append(
                {"surface": raw, "surface_norm": sn, "graph_id": gid,
                 "curie": disp, "label": DISEASE_LABEL, "src": src}
            )

    dic = pl.DataFrame(
        records,
        schema={"surface": pl.Utf8, "surface_norm": pl.Utf8, "graph_id": pl.Utf8,
                "curie": pl.Utf8, "label": pl.Utf8, "src": pl.Utf8},
    ).unique(subset=["surface_norm", "graph_id"])  # one row per (surface, node)

    dic.write_parquet(OUT_PARQUET)

    # --- diagnostics ---
    by_src = {r["src"]: r["len"] for r in dic.group_by("src").len().iter_rows(named=True)}
    distinct_nodes = dic.select("graph_id").n_unique()
    amb = (
        dic.group_by("surface_norm")
        .agg(pl.col("graph_id").n_unique().alias("n_nodes"))
        .filter(pl.col("n_nodes") > 1)
        .height
    )
    stats = {
        "source": str(SRC),
        "dis_nodes_total": n_dis_all,
        "dis_nodes_after_oba_drop": n_dis_kept,
        "oba_dropped": n_dis_all - n_dis_kept,
        "strict_attr_drop": STRICT_ATTR_DROP,
        "surface_keys_kept": list(active_keys.keys()) + ["name"],
        "scope_keys_excluded": list(EXCLUDED_SCOPE_KEYS),
        "excluded_surfaces_skipped": excluded_skipped,
        "surface_rows": dic.height,
        "surfaces_by_src": by_src,
        "properties_json_parse_errors": n_blob_err,
        "distinct_disease_nodes": distinct_nodes,
        "ambiguous_surface_forms": amb,
        "norm_column_used": norm_col,
        "output": str(OUT_PARQUET),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))

    print("\n--- B1 summary ---")
    print(f"DIS nodes: {n_dis_all} -> {n_dis_kept} after OBA drop "
          f"({n_dis_all - n_dis_kept} OBA removed)")
    print(f"surface rows: {dic.height}  by src: {by_src}")
    print(f"excluded (broad/narrow/related) surfaces skipped: {excluded_skipped}")
    print(f"distinct disease nodes: {distinct_nodes}")
    print(f"ambiguous surfaces (same string -> multiple nodes): {amb}")
    print("prefix distribution of kept DIS nodes:")
    print(prefix_dist)
    print(f"wrote {OUT_PARQUET}")
    print(f"wrote {OUT_STATS}")


if __name__ == "__main__":
    main()
