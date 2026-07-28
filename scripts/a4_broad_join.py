#!/usr/bin/env python3
"""
a4_broad_join.py — rebuild the Broad browser's combined view from the two bulk files.

There is NO public API. Broad's FAQ is explicit: "To download repurposing annotations
in batch format, the Drug information and Sample information downloadable files have
relevant information and you do not need an API." The 2017 paper's JSON API was the
clue.io-era app (registration + key), not an endpoint on repo-hub.

The browser's table is a JOIN, not a third file:
    drug annotation  (7,540 rows, 1 per compound) : pert_iname, clinical_phase, moa,
                                                    target, disease_area, indication
    sample annotation (22,612 rows, ~3 per compound): broad_id, pert_iname, ..., smiles,
                                                    InChIKey, pubchem_cid, vendor, ...
    joined on pert_iname.

Every column in the browser's SHOW/HIDE list comes from one of these two, except
"Is Plated?" which is library-membership state and exists only in the app.

THE TRAP this script exists to avoid: the sample file is PHYSICAL-SAMPLE level. A naive
left join multiplies each compound by its sample count (2,718 Launched drugs became
9,009 rows in an earlier pass, and every percentage computed off it was wrong). And 148
compounds ship as >1 salt/stereo form with DIFFERENT InChIKeys, so `.unique(pert_iname)`
picks one by row order. This script aggregates per compound and KEEPS every key, with an
explicit n_inchikeys column, so the choice is made by A6 with a stated rule rather than
inherited from file order.

Read-only on inputs. Writes data/canon/broad_compounds.parquet (one row per compound).

    python scripts/a4_broad_join.py \
        --drug   data/sources/repo-drug-annotation-20200324.txt \
        --sample data/sources/repo-sample-annotation-20240610.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import polars as pl

REPO = Path(os.environ.get("RETRACERX_ROOT", ".")).resolve()
OUT = REPO / "data" / "canon" / "broad_compounds.parquet"
MANIFEST = REPO / "data" / "manifest" / "broad_manifest.json"


def read_broad(path: Path) -> tuple[pl.DataFrame, dict]:
    """'!' lines are TSV rows padded with tabs to the column count — strip everything."""
    hdr = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("!"):
            break
        m = re.match(r"^!(\S+)\s*(.*)$", line)
        if m:
            hdr[m.group(1)] = m.group(2).strip().strip('"')
    df = pl.read_csv(path, separator="\t", comment_prefix="!",
                     infer_schema_length=10_000, truncate_ragged_lines=True)
    return df, hdr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    drug, dhdr = read_broad(args.drug)
    samp, shdr = read_broad(args.sample)
    print(f"drug   : {drug.height:,} rows, !File_date={dhdr.get('File_date')!r}")
    print(f"sample : {samp.height:,} rows, !File_date={shdr.get('File_date')!r}")

    # The filename is not the version; the header is. Two files, two independent
    # refreshes — nothing guarantees they move together, and they join on pert_iname.
    if dhdr.get("File_date") != shdr.get("File_date"):
        raise SystemExit(
            f"FATAL: different vintages — drug={dhdr.get('File_date')!r} "
            f"sample={shdr.get('File_date')!r}. They join on pert_iname; a mismatched "
            f"pair drops compounds silently and reads as missing InChIKeys.")

    n_drug_rows, n_drug_cmpd = drug.height, drug["pert_iname"].n_unique()
    if n_drug_rows != n_drug_cmpd:
        print(f"NOTE: drug file is not 1-row-per-compound: {n_drug_rows:,} rows / "
              f"{n_drug_cmpd:,} pert_iname — deduping.")
        drug = drug.unique("pert_iname", keep="first")

    ik = next(c for c in samp.columns if c.lower() in ("inchikey", "inchi_key"))

    def uniq(col: str, alias: str) -> pl.Expr:
        return (pl.col(col).cast(pl.Utf8).str.strip_chars()
                .filter(pl.col(col).is_not_null() & (pl.col(col).cast(pl.Utf8) != ""))
                .unique().alias(alias))

    # Aggregate to COMPOUND level first. This is the whole point: join 1:1 afterwards.
    agg = [pl.len().alias("n_samples"), uniq(ik, "inchikeys"), uniq("smiles", "smiles_list")]
    for src, dst in (("broad_id", "broad_ids"), ("pubchem_cid", "pubchem_cids"),
                     ("vendor_name", "vendors"), ("deprecated_broad_id", "deprecated_ids")):
        if src in samp.columns:
            agg.append(uniq(src, dst))
    sagg = samp.group_by("pert_iname").agg(agg)

    j = drug.join(sagg, on="pert_iname", how="left")
    assert j.height == drug.height, "join multiplied rows — sample aggregation failed"

    out = j.with_columns(
        pl.col("inchikeys").list.len().fill_null(0).alias("n_inchikeys"),
        # Broad pipe-separates multi-target rows. A4's gene bridge needs them exploded,
        # so surface the list rather than making A4 re-parse the string.
        pl.col("target").str.split("|").alias("targets"),
    ).with_columns(
        # ONE key only when it is unambiguous. Ambiguity is reported, never resolved here.
        pl.when(pl.col("n_inchikeys") == 1).then(pl.col("inchikeys").list.first())
          .otherwise(None).alias("inchi_key"),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out, compression="zstd")

    n = out.height
    launched = out.filter(pl.col("clinical_phase") == "Launched")
    multi = out.filter(pl.col("n_inchikeys") > 1)
    print(f"\nwrote {args.out} — {n:,} compounds, {out.width} columns")
    print(f"  columns: {out.columns}")
    print(f"\n  with >=1 InChIKey        : {out.filter(pl.col('n_inchikeys') > 0).height:,}/{n:,}")
    print(f"  with NO sample row       : {out.filter(pl.col('n_samples').is_null()).height:,}"
          f"   <- in the drug file, absent from the sample file: no key obtainable")
    print(f"  AMBIGUOUS (>1 InChIKey)  : {multi.height:,}   <- inchi_key is null for these"
          f" BY DESIGN; A6 states the rule")
    print(f"  Launched compounds       : {launched.height:,}, "
          f"{launched.filter(pl.col('n_inchikeys') > 0).height:,} with a key")

    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
        if man.get("file_date") not in (None, dhdr.get("File_date")):
            print(f"\n  WARNING: broad_manifest.json pins File_date={man['file_date']!r} but "
                  f"these files say {dhdr.get('File_date')!r}. Re-pin before trusting.")
    else:
        print(f"\n  WARNING: {MANIFEST} missing — these bytes are unpinned. Broad refreshes "
              f"content behind stable URLs, so sha256 + !File_date are the only version.")


if __name__ == "__main__":
    main()
