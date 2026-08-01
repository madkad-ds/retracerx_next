# ReTraceRx — Retrieval Contract v1.0

**Status:** FROZEN (shape). **Direction:** disease-in / drug-candidates-out.
**Producer:** Phase B (`retrieve_candidates`). **Consumer:** Phase C (agent + faithfulness verifier), Phase D harness.

This is an **interface contract**, not an output format to be discovered. B emits exactly this;
C is built against it. The *shape* is fixed at v1.0 — iterate contents (ranking, path counts, K,
evidence phrasing) freely, and bump `contract_version` only if a field is added, removed, renamed,
or changes type.

---

## Shape (annotated example)

```json
{
  "contract_version": "1.0",
  "query_disease": {
    "text": "type 2 diabetes",          // the raw user string, verbatim
    "graph_id": "MONDO_0005148",         // Neo4j key (verbatim id) — what B3 Cypher matches on
    "curie": "MONDO:0005148",            // colon-form CURIE — DISPLAY ONLY
    "link_score": 0.91,                  // B2 linker confidence in [0,1]
    "alternatives": []                   // populated only when link confidence is low (see below)
  },
  "candidates": [
    {
      "rank": 1,                         // 1-based; mirrors sort order exactly
      "drug_graph_id": "CHEMBL_1431",    // Neo4j key (verbatim id)
      "drug_curie": "CHEMBL:1431",       // display only
      "drug_name": "metformin",
      "score": 0.79,                     // B4 composite — THE sort key
      "kge_score": 0.83,                 // primary signal; NULLABLE (see semantics)
      "support_count": 3,                // # independent paths (B4 component)
      "specificity": 0.6,                // path-quality term (B4 component)
      "paths": [
        {
          "path_id": "p_9f2a1b3c4d",     // stable hash of the ordered triples
          "triples": [
            ["CHEMBL_1431", "AFFECTS", "NCBIGene_5562"],
            ["NCBIGene_5562", "GENE_ASSOCIATED_WITH_CONDITION", "MONDO_0005148"]
          ],
          "nl_evidence": "Metformin affects PRKAA1, which is associated with type 2 diabetes.",
          "sources": ["DGIdb", "OpenTargets"]
        }
      ]
    }
  ]
}
```

---

## Frozen semantics

| Field | Rule |
|---|---|
| `graph_id` / `drug_graph_id` | The **verbatim** OptimusKG `id` (underscore form). The only key Neo4j and B3 match on. Never substitute the display CURIE here. |
| `curie` / `drug_curie` | Colon-form CURIE for display/citation only. Never used as a graph key. |
| `candidates` order | **Pre-sorted by `score` descending.** Consumers must not re-sort to interpret rank. |
| `rank` | 1-based, mirrors the sorted order. `rank == index+1`. |
| `score` | The B4 composite (`0.6·kge + 0.3·min(support,5)/5 + 0.1·specificity`, current MVP weights). This is the sort key. |
| `kge_score` | Primary signal but **nullable**. `null` = the drug is unscoreable by the KGE (A8 reachability / transductive missing-entity). **Never coerce `null` to 0** — 0 silently sinks a candidate; `null` says "no score exists." A `null`-KGE candidate can still rank on path support. |
| `support_count`, `specificity` | B4 components, surfaced for reproducibility (B4: log the exact scoring). |
| `path_id` → `triples` | Deterministic, exact, stable. The Phase C faithfulness verifier **re-derives `path_id` from `triples`** to confirm the model didn't cite a fabricated path. Do not change the hashing without a version bump. |
| `triples` | Ordered list of `[head_graph_id, RELATION, tail_graph_id]`, all verbatim graph ids, RELATION = the composite `LABEL__RELATION` type (50 surviving types, A3 Step 5). |
| `sources` | **Always a list** (matches A2 `sources_direct`), even with one element. Carried for citation. |
| `alternatives` | Empty by default. When `link_score` is low, B2 returns competing `{text, graph_id, curie, link_score}` objects here so the agent can disambiguate instead of the retriever silently linking wrong. |

---

## JSON Schema (for programmatic validation)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ReTraceRx Retrieval Contract v1.0",
  "type": "object",
  "required": ["contract_version", "query_disease", "candidates"],
  "properties": {
    "contract_version": {"const": "1.0"},
    "query_disease": {
      "type": "object",
      "required": ["text", "graph_id", "curie", "link_score", "alternatives"],
      "properties": {
        "text": {"type": "string"},
        "graph_id": {"type": "string"},
        "curie": {"type": "string"},
        "link_score": {"type": "number", "minimum": 0, "maximum": 1},
        "alternatives": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["text", "graph_id", "curie", "link_score"],
            "properties": {
              "text": {"type": "string"},
              "graph_id": {"type": "string"},
              "curie": {"type": "string"},
              "link_score": {"type": "number", "minimum": 0, "maximum": 1}
            }
          }
        }
      }
    },
    "candidates": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["rank", "drug_graph_id", "drug_curie", "drug_name",
                     "score", "kge_score", "support_count", "specificity", "paths"],
        "properties": {
          "rank": {"type": "integer", "minimum": 1},
          "drug_graph_id": {"type": "string"},
          "drug_curie": {"type": "string"},
          "drug_name": {"type": ["string", "null"]},
          "score": {"type": "number"},
          "kge_score": {"type": ["number", "null"]},
          "support_count": {"type": "integer", "minimum": 0},
          "specificity": {"type": "number"},
          "paths": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["path_id", "triples", "nl_evidence", "sources"],
              "properties": {
                "path_id": {"type": "string"},
                "triples": {
                  "type": "array",
                  "items": {
                    "type": "array",
                    "minItems": 3, "maxItems": 3,
                    "items": {"type": "string"}
                  }
                },
                "nl_evidence": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}}
              }
            }
          }
        }
      }
    }
  }
}
```

---

## Versioning policy

- **v1.0 shape is frozen.** C pins to `contract_version == "1.0"`.
- Content tuning (weights, K, M paths/candidate, phrasing) does **not** change the version.
- Any field add/remove/rename/type-change → new version (`1.1`, `2.0`) + a migration note here.
- The typed / bidirectional generalization (drug-in *or* disease-in) is a **known future v2.0**, deliberately deferred; do not retrofit it into v1.0.

## Change log

| version | date | change |
|---|---|---|
| 1.0 | (freeze date) | Initial frozen shape. Adds `contract_version`, `link_score`/`alternatives`, composite `score`+`rank`, nullable `kge_score`, `graph_id`/`curie` split, list-valued `sources` over the how-to draft. |
