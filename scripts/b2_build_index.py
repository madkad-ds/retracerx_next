#!/usr/bin/env python3
"""
ReTraceRx Phase B — B2.1: build & persist the disease-linking vector index.

Encodes every surface form in B1's link_dict_disease.parquet with SapBERT
(bi-encoder; [CLS] embedding, L2-normalized so inner product == cosine) and
writes a FAISS IndexFlatIP plus a ROW-ALIGNED metadata parquet: index vector i
corresponds to meta row i. Run once; rerun only when the B1 dictionary changes.

Outputs:
  data/retrieval/faiss_disease.index        (FAISS index over surface embeddings)
  data/retrieval/faiss_disease_meta.parquet (surface/surface_norm/graph_id/curie/src, add-order)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

DICT = Path("data/retrieval/link_dict_disease.parquet")
IDX = Path("data/retrieval/faiss_disease.index")
META = Path("data/retrieval/faiss_disease_meta.parquet")
MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DIM = 768
BATCH = 256
MAXLEN = 32  # disease mentions are short; SapBERT convention


def embed_all(surfaces: list[str], tok, enc, dev: str) -> np.ndarray:
    vecs = np.zeros((len(surfaces), DIM), dtype="float32")
    with torch.no_grad():
        for i in range(0, len(surfaces), BATCH):
            batch = surfaces[i:i + BATCH]
            t = tok(batch, padding=True, truncation=True, max_length=MAXLEN,
                    return_tensors="pt").to(dev)
            v = enc(**t).last_hidden_state[:, 0, :]              # [CLS]
            v = torch.nn.functional.normalize(v, dim=1)          # cosine via IP
            vecs[i:i + len(batch)] = v.cpu().numpy().astype("float32")
            if (i // BATCH) % 50 == 0:
                print(f"  encoded {i}/{len(surfaces)}")
    return vecs


def main():
    if not DICT.exists():
        raise FileNotFoundError(f"{DICT} missing — run b1_build_link_dict.py first.")
    df = pl.read_parquet(DICT)                 # THIS order becomes the index order
    surfaces = df["surface"].to_list()         # encode raw surface; SapBERT handles casing

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  surfaces={len(surfaces)}  model={MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = AutoModel.from_pretrained(MODEL).to(dev).eval()

    vecs = embed_all(surfaces, tok, enc, dev)
    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)

    IDX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(IDX))
    df.write_parquet(META)                     # aligned 1:1 to add order

    assert index.ntotal == df.height, "index/meta length mismatch"
    print(f"wrote {IDX} (ntotal={index.ntotal}) and {META} (rows={df.height})")


if __name__ == "__main__":
    main()
