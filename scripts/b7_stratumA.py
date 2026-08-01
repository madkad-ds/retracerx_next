"""Stratum-A (graph-independent) recall from the B7 per-query CSV.
Graph-independent = held-out positive whose (drug,disease) was NOT a removed
DRG-DIS indication edge (the ~94.5% circular ones were; A5/A7)."""
import polars as pl
from pathlib import Path

pq = pl.read_csv("data/retrieval/b7_disease_cold_perquery.csv")

# removed indication edges: from,to,label,relation,endpoint_key  (U+241F sep)
rem = pl.read_csv("data/holdout/removed_edge_ids.csv")
print("removed_edge cols:", rem.columns)
# keep only DRG-DIS indication rows; build the set of {drug,disease} unordered pairs
lab = "label" if "label" in rem.columns else None
ind = rem.filter(pl.col(lab).str.contains("DRG-DIS")) if lab else rem
# from/to are the two endpoints; a positive is circular if its (drug,disease) pair
# appears as a removed edge in either orientation
pairs = set()
fcol = "from" if "from" in ind.columns else ind.columns[0]
tcol = "to" if "to" in ind.columns else ind.columns[1]
for a, b in ind.select([fcol, tcol]).iter_rows():
    pairs.add(frozenset((a, b)))

def is_circular(drug, disease):
    return frozenset((drug, disease)) in pairs

pq = pq.with_columns(
    pl.struct(["drug", "disease"]).map_elements(
        lambda s: is_circular(s["drug"], s["disease"]), return_dtype=pl.Boolean
    ).alias("circular")
)

def recall(df, K):
    if df.height == 0: return None
    return round((df["rank"].is_not_null() & (df["rank"] <= K)).sum() / df.height, 4)

for label, sub in [("ALL", pq),
                   ("Stratum A (graph-independent)", pq.filter(~pl.col("circular"))),
                   ("Stratum A ∩ reachable", pq.filter(~pl.col("circular") & pl.col("reachable")))]:
    n = sub.height
    pool = round(sub["in_pool"].sum()/n, 4) if n else None
    print(f"\n{label}: n={n}")
    print(f"  pool_recall={pool}  recall@10={recall(sub,10)}  recall@20={recall(sub,20)}  recall@50={recall(sub,50)}")
