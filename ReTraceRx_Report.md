# ReTraceRx: A Faithfulness Constrained, Self Hosted Graph-RAG Agent for Drug Repurposing

**Course:** LLMs - A Hands-on Approach (May – July 2026), CCE IISc
**Team Members:** Madhura Kadaba (madhura.kadaba@gmail.com)
**Date:** 9 August 2026
*version 1.0*

---

## Part I: Overview and Context

### 1. Abstract

ReTraceRx is a self-hosted, faithfulness constrained Graph-RAG drug repurposing Agent built on a rigorously leakage controlled biomedical knowledge graph. Its current contribution is _not_ beating baselines on recovery of novel drug–disease indications; that signal is structurally near-zero for *every* system at 2-hop retrieval, and we report it as such. The contribution is four fold: (1) a provenance tracked data foundation that **measures its own 94.5% circularity and removes the entire drug–disease relation to prevent leakage**; (2) a grounded agent that **eliminates fabricated citations: 0 across ~7,300 picks** against an ungrounded LLM baseline whose pick level hallucination is 1.0 by construction; (3) **calibrated abstention**, in which the agent declines rather than invents (confident invention = 0; confidence separates faithful from unfaithful picks at AUROC ≈ 0.99); and (4) a **precise, reproducible diagnosis of why 2-hop retrieval is blind to novel repurposing**, which scopes 3-hop retrieval as the concrete next lever.

---

### 2. Problem statement & Relevance

Large language models fabricate plausible sounding drug mechanisms with high confidence, which is disqualifying in a domain where an unverifiable claim is worse than no claim. Grounded, retrieval-augmented systems mitigate this, but the strongest published systems are typically API-bound (dependent on commercial frontier models) and rarely constrain the model such that fabrication is structurally impossible rather than merely discouraged. Drug repurposing i.e. finding new indications for existing drugs is a high impact application because the safety profile of the molecule is already established, so the bottleneck is candidate generation and trustworthy evidence.

This project is a study in *faithfulness and grounding* (can we make citations un-fakeable?), *agentic retrieval over structured graphs* (can an agent reason over a typed knowledge graph and cite its evidence?), and *self-hosted serving* (can the whole system run without a commercial API dependency?). ReTraceRx is able to achieve all three, but currently is *not* able to recover *novel* indications that lie beyond the 2-hop neighbourhood.

---

### 3. Approach & Architecture

ReTraceRx is a five stage pipeline assembled into a single system:

**Data → Retrieval → KGE baseline → Grounded agent → Evaluation → Deployment.**

1. **Data foundation.** A filtered biomedical knowledge graph (OptimusKG-2.0: ~190,531 nodes, ~21.7M edges, 50 composite relations) is loaded into Neo4j after aggressive leakage control and provenance tracking. Frozen train/test holdouts are cut under three regimes (`random`, `disease_cold` = unseen disease, `drug_cold` = unseen drug), seed `20260727`.

2. **Retrieval (Graph-RAG).** A SapBERT/FAISS entity linker resolves a free-text disease to a graph node; a 2-hop metapath retriever walks `disease —DIS_GEN__ASSOCIATED_WITH→ gene —DRG_GEN__{TARGET,…}→ drug` to assemble candidate drugs and their supporting paths; a ranker scores them; and a serializer emits a versioned retrieval contract (v1.0) carrying, for every candidate, its paths as `(path_id, triples, nl_evidence, sources)`.

3. **KGE baseline.** A per split RotatE knowledge graph embedding model (trained on the leakage controlled graph plus train split indications only) provides the link prediction floor against which the agent is measured.

4. **Grounded agent.** A LangGraph agent (Qwen/Qwen3.5-9B, FP16, served on vLLM) reasons per candidate and emits citations that are enum-constrained to the retrieved `path_id`s via guided JSON decoding — so a fabricated citation is structurally impossible at generation time. A verifier re-checks every citation against the graph as defence in depth.

5. **Evaluation & deployment.** A single metric engine with pluggable rankers produces a circularity stratified ablation matrix with bootstrap and paired confidence intervals; the whole stack deploys as two containers (Neo4j + vLLM) plus a host application (an evidence-subgraph explorer).

The **faithfulness mechanism** is centered in the architecture: because the model may only cite `path_id`s that appear in the retrieved contract (enforced by a guided decoding enum, not a post-hoc filter), and because those paths are deterministically serialized from real graph edges, a citation cannot point at anything that does not exist in the graph.

![ReTraceRx system architecture: the five-phase pipeline from data foundations through retrieval, the grounded agent with its enum-constrained faithfulness mechanism, evaluation, and deployment.](https://github.com/madkad-ds/retracerx/blob/7a8e21f8e19aa56361cc056b5d5776e60dbbaffb/ReTraceRx_architecture_simple_clean.svg)

---

### 4. Related work & Novelty

ReTraceRx sits alongside recent graph grounded LLM systems for biomedical reasoning, among them **CLADD** (retrieval augmented, agentic drug reasoning), **K-Paths** (path based knowledge retrieval), **ESCARGOT** (a Graph-of-Thoughts approach), and **TxGNN** (graph neural networks for zero shot drug indication prediction).

ReTraceRx's differentiators are:

- **Self-hosted and fine-tuning-free:** The entire system runs on a single GPU with an open weights model served on vLLM; no commercial API and no task specific fine tuning are required.
- **Enum constrained, un-fakeable citations:** Faithfulness is enforced structurally at decode time rather than by prompting or post-hoc filtering.
- **Calibrated abstention:** The agent declines when it cannot verify support, and its confidence is calibrated to that support decision.
- **True-negative discrimination:** Evaluation uses frozen, resolved repoDB-derived negatives, not only positives, so specificity is measurable rather than assumed.

---
---

## Part II: System & Evidence

![ReTraceRx system architecture: the five-phase pipeline from data foundations through retrieval, the grounded agent with its enum-constrained faithfulness mechanism, evaluation, and deployment.](https://github.com/madkad-ds/retracerx/blob/7a8e21f8e19aa56361cc056b5d5776e60dbbaffb/ReTraceRx_architecture.svg)

### 5. Data foundations (Phase A)

The knowledge substrate is **OptimusKG-2.0**: ~190,531 nodes, ~21.7M edges, and 50 composite relation types. Gold indications and negatives derive from repoDB (8,390 positive and 2,212 negative drug–disease pairs; 3,943 positives and 1,425 negatives are evaluable after entity resolution). Entity resolution keys on **InChIKey** (drugs) and **MONDO** (diseases), with a reachable candidate whitelist of **1,565** drugs and full crosswalks retained for recovery matching.

*Leakage control is the defining discipline of Phase A.* The entire drug–disease relation label was removed from the retrieval graph **70,380 edges** so that neither the retriever nor the agent can "recover" an indication by reading it directly. The KGE baseline is then given back *train-split-only* indications (reconstructed from the removed edge ledger and collapse normalized before the split), so the embedding model learns the relation while test pairs remain unseen; the agent's retrieval graph stays fully stripped. Neither system sees *test* pairs.

Three frozen holdouts (`random`, `drug_cold`, `disease_cold`; seed `20260727`) support the ablation, with per split negatives held out from the same frozen bundle.

The single most important measurement in the data foundation is circularity: **94.5%** of evaluable positives are *circular*, the held-out drug is reachable from the held-out disease through a shared gene that itself encodes the (removed) indication's biology, so the "prediction" is partly tautological. Only **~217** positives are **graph-independent (Stratum A)**, genuinely novel predictions that do not rest on a shared disease-gene. This stratum, not the full set, is where novel repurposing is actually tested. A companion boundary is the **~49% disease-coverage ceiling**: only roughly half of held-out diseases are representable in the retrieval graph.

Measuring and reporting these ceilings helped in interpreting every downstream number.

---

### 6. Retrieval (Phase B)

The linker resolves free-text disease mentions to graph nodes via a three layer escalation (exact → fuzzy → SapBERT/FAISS embedding over an 83,347 surface dictionary), returning the full `query_disease` contract block with a link score and, on low confidence matches, a disambiguation set rather than a silent wrong link. The 2-hop retriever walks the metapath `disease —DIS_GEN__ASSOCIATED_WITH→ gene —DRG_GEN__{TARGET,…}→ drug`, with **560 hub nodes blacklisted** to prevent promiscuous genes from flooding the candidate pool. The serializer emits contract v1.0 with deterministic `path_id`s.

Naive 3-hop retrieval was attempted and is intractable on this graph, a third hop fans out combinatorially through the ~1,901 diseases with >1,000 gene associations, timing out past 90s. The per-query "fall back to 2-hop only when 3-hop is too big" workaround was ruled out because the dense diseases are the majority, not the exception.

2-hop retrieval result:

- **Pool recall ≈ 0.20–0.25** overall — the retriever surfaces the true drug into its candidate pool for roughly a fifth to a quarter of queries.
- **Stratum-A pool recall = 0.0 on all three splits.**

A Stratum-A (graph independent) positive is *defined* by the true drug **not** sharing a direct disease-gene with the disease. The 2-hop metapath is exactly "disease → shared gene → drug"; a candidate that has no shared disease-gene is not reachable in two hops. So the retriever cannot pool a novel candidate whose novelty consists precisely of lacking the 2-hop connection. This is the ceiling that bounds every recovery number downstream, and it is the motivation for 3-hop retrieval (§11).

---

### 7. Agent & faithfulness (Phase C)

The grounded agent is a LangGraph pipeline (link → retrieve → reason → verify → assemble) running Qwen/Qwen3.5-9B in FP16 on vLLM. Per candidate, it emits a decision whose `cited_path_ids` are **enum-constrained** by guided JSON decoding to the retrieved `path_id`s. A verifier provides defence in depth: it re-derives each `path_id` from its triples using the shared `compute_path_id` (never reimplemented), checks membership in the contract, and performs a sampled undirected Neo4j re-check that each cited edge is a real graph edge.

**Faithfulness — the headline result.**

| split | grounded pick-hallucination (95% CI) | ungrounded pick-hallucination | hallucination gap | `citation_failure` |
|---|---|---|---|---|
| disease_cold | 0.2465 [0.2062, 0.2887] | 1.0 *(by construction)* | 0.7535 | **0** |
| random | *(within 0.74–0.77 band)* | 1.0 | ~0.74–0.77 | **0** |
| drug_cold | *(within 0.74–0.77 band)* | 1.0 | ~0.74–0.77 | **0** |

Across *~7,300 picks on the full holdout, `citation_failure = 0`* every citation the grounded agent ever makes verifies against the graph. The ungrounded baseline saw no graph and therefore has no path to cite, so its pick level hallucination is *1.0 by construction*; the resulting **hallucination gap is 0.74–0.77**.

Two points of rigour: 
First, the grounded "unfaithful" rate (~0.25) is **not fabrication** it decomposes into `unsupported` (the agent declined to claim adequate support, a *calibration* signal) and `citation_failure` (a verifier-catch of a fabricated citation). The latter is **0**; the ~0.25 is entirely the former. The headline is therefore "0 fabricated citations," not "0.25 unfaithful." 
Second, the ungrounded 1.0 is a definitional floor, and we present it as such rather than as an empirical defeat of a strong baseline.

**Calibration & abstention.**

| split | abstain rate | confident-invention | AUROC(confidence → faithful) | mean conf faithful / unfaithful |
|---|---|---|---|---|
| disease_cold | 0.0224 | **0** | 0.9924 | 0.756 / 0.101 |
| random | 0.0576 | **0** | 0.9879 | 0.733 / 0.119 |
| drug_cold | 0.0595 | **0** | 0.9900 | 0.737 / 0.116 |

The agent abstains on **2–6%** of diseases (cause: zero reachable candidate, or no faithfully-supportable pick) and **never confidently invents** (confident-invention = 0 on every split). Its stated confidence separates faithful from unfaithful picks at **AUROC ≈ 0.99**, and reliability is monotone (low-confidence bins are ~0% faithful; high-confidence bins ~100%).

*Calibration caveat* Under guided decoding, "faithful" ≈ "supported," and the confidence is emitted alongside the support decision — so this AUROC measures the agent's **self-consistency** with its own support call, not calibration to ground-truth clinical correctness. We report it as self-consistency, not as external validity.

---

### 8. Baseline & evaluation (Phase D)

Evaluation runs through **one metric engine with pluggable rankers**: the harness owns the holdout, whitelist, circularity strata, filtering, and ceilings, and each system (KGE, grounded agent, ungrounded LLM, graph-RAG, mock) only supplies `rank(disease, universe) → best-first drug ids`. Freezing this interface structurally guarantees that every system is measured on identical data — a ranker cannot diverge the metric — and the KGE ranker reproduces the standalone D3 evaluation, proving the harness is faithful. Two candidate modes are emitted and *never averaged*: `plus_true` (capability — rank the true drug among 1,565 distractors) and `whitelist` (deployment — recover from a fixed shortlist). Stratum A (graph-independent) is primary.

**Recovery is retrieval-bound.** On the primary Stratum-A `plus_true` cells, MRR is near-zero for every system and the confidence intervals overlap heavily, no system separates from any other. Point estimates with 95% bootstrap CIs (2,000 resamples, seed `20260727`):

| system | disease_cold (n=31) | random (n=38) | drug_cold (n=24) |
|---|---|---|---|
| grounded agent | 0.0018 [0.0012, 0.0026] | 0.0285 [0.0018, 0.0814] | 0.0051 [0.0024, 0.0095] |
| KGE (RotatE) | 0.0036 [0.0015, 0.0066] | 0.0282 [0.0045, 0.0612] | 0.0111 [0.0028, 0.0226] |
| graph-RAG | 0.0019 [0.0013, 0.0027] | 0.0110 [0.0019, 0.0287] | 0.0052 [0.0025, 0.0091] |
| ungrounded LLM | 0.0020 [0.0013, 0.0029] | 0.0287 [0.0020, 0.0816] | 0.0082 [0.0027, 0.0181] |
| mock (chance floor) | — | 0.0040 [0.0020, 0.0068] | — |

Every cell sits in the same near-zero band, and on `random` the CIs are so wide (n=38, e.g. agent [0.0018, 0.0814]) that the systems are formally inseparable. Stratum-A **Hits@10** is at or near zero throughout: KGE gives 0.00 (disease_cold), 0.0526 (random), 0.0417 (drug_cold); the grounded agent gives 0.00 / 0.0263 / 0.00 — and on the zero-shot `disease_cold` split every system's Hits@10 is exactly 0. *(Per the CI-width discipline, whitelist Stratum-A cells, n=4–6, are never quoted as point estimates.)*

**Paired significance (agent − baseline ΔMRR, Stratum-A, plus_true).** 
Resampling the same query indices for both systems isolates the system effect:

| split | baseline | n | ΔMRR | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|
| disease_cold | KGE | 31 | −0.0018 | [−0.0048, +0.0005] | no |
| disease_cold | graph-RAG | 31 | −0.0001 | [−0.0002, +0.0000] | no |
| disease_cold | ungrounded | 31 | −0.0002 | [−0.0003, −0.0001] | yes° |
| random | KGE | 38 | +0.0003 | [−0.0287, +0.0354] | no |
| random | graph-RAG | 38 | +0.0175 | [−0.0001, +0.0526] | no |
| random | ungrounded | 38 | −0.0001 | [−0.0002, −0.0001] | yes° |
| drug_cold | KGE | 24 | −0.0060 | [−0.0184, +0.0039] | no |

Every non-degenerate CI includes 0, the agent is *statistically indistinguishable from every baseline on novel recovery*. The two cells marked *yes°* exclude 0 only because the delta is degenerate (|ΔMRR| ≈ 0.0001–0.0002); they are not a meaningful recovery advantage in either direction. (Per-cell MRR CIs for all five systems × three splits are tabulated in `d7_percell_ci.csv` and shown in the matrix above; they overlap across systems in every cell.) 

**Ablation Results: at 2-hop, no system recovers novel indications, and the grounded agent neither beats nor is beaten by the floor.**

**True-negative discrimination (repoDB):** Against the frozen, resolved per-split holdout negatives (random 319, disease_cold 208, drug_cold 323), the KGE link prediction model's discrimination is **at chance on every split**. On the honest graph independent positives (`A_only`), AUROC is 0.4435 [0.3326, 0.556] (disease_cold), 0.4465 [0.3596, 0.5379] (random), and 0.5396 [0.4058, 0.665] (drug_cold); on the full reachable set (`full_AB`) it is 0.4767 [0.4356, 0.5189], 0.4370 [0.3977, 0.4759], and 0.5076 [0.4726, 0.5454] respectively. **Every one of these six 95% CIs includes 0.5.** Because the reachable holdouts are positive-heavy (prevalence 0.69–0.78), AUPRC is interpreted only against its prevalence baseline; on the `A_only` positives (prevalence 0.07–0.13) AUPRC is 0.18 / 0.09 / 0.19 at or below prevalence, i.e. no real precision signal either.

**Error taxonomy:** Misses decompose into a structural ceiling (unreachable pairs) and ranking misses (all reachable misses), with the agent adding an explicit **abstention** bucket (2–6%). For KGE, retrieval/reasoning/hallucination buckets are N/A by construction (it ranks the entire reachable pool), so 100% of its reachable errors are ranking misses.

**Recovery is structurally ceiling-bound at 2-hop; every low cell is reported with its cause and its confidence interval, and both candidate modes and all three splits are shown rather than a favourable subset.**

---

### 9. Implementation & reproducibility

Reproducibility is proven by construction:

- **Pinned everything.** Per-phase lockfiles, fixed seeds (`20260727` for splits and bootstraps; KGE training seed `42`), and a digest-pinned vLLM image (`sha256`) so the serving environment cannot drift.
- **Gates.** Formal gate checks at each phase boundary (e.g. the Phase-D KGE-data assembly gate passes 4/4 leakage/vocab/reconciliation/collapse assertions), and a deploy preflight that verifies every consumed artifact and pin before the stack comes up.
- **Harness integrity.** One metric engine, pluggable rankers; the KGE ranker reproduces the standalone evaluation to the digit, which is the structural proof that all systems are measured identically.
- **Determinism.** Greedy decoding plus a fixed seed yield byte-identical agent replays across invocations.
- **One-command deployment.** The runtime stack is two containers (Neo4j + vLLM) plus a host application launched from a pinned virtual environment; the repository ships the pipeline and scripts (not bulk data), with attribution/NOTICE for upstream sources.

---

### 10. Deployment & demo (Phase E)

The system deploys as a **two-container stack** (Neo4j with the leakage-filtered graph and APOC; vLLM serving Qwen3.5-9B in FP16 under a pinned digest) plus a **host application** that reaches the containers over published local ports. The application has two faces:

- An **evidence-subgraph explorer**: enter a disease, get a ranked candidate shortlist, and for each pick see its cited paths rendered as an interactive subgraph beside the citable triples and their sources. Low-confidence links surface a disambiguation panel rather than guessing, and abstentions render as a first-class outcome.
- A **grounded-vs-ungrounded contrast**: the same disease answered with and without the graph, making the faithfulness gap visible — the grounded agent cites verifiable paths or abstains, while the ungrounded baseline answers confidently with zero citations.

---
---

## Part III: Assessment & Outlook

### 11. Limitations & future work

- **The 2-hop Stratum-A ceiling is the binding limitation.** Novel (graph independent) candidates lack a shared disease-gene by definition and are therefore unreachable by the "disease → gene → drug" metapath. This is why Stratum-A recovery is ~0 for every system, and it is a property of the retrieval depth, not of any ranker.
- **3-hop retrieval is the scoped lever.** Extending the metapath by one hop (e.g. disease → gene → pathway/gene → drug) is the concrete next step expected to move novel recovery off zero.
- **Coverage and circularity boundaries.** The ~49% disease-coverage ceiling and the 94.5% circularity of the evaluable set bound what any 2-hop system on this graph can be asked to do; both are measured and reported rather than hidden.

The faithfulness and calibration machinery is complete and reusable, and 3-hop retrieval is the single change most likely to convert the negative recovery result into a positive one.

---

### 12. Conclusion

ReTraceRx is a faithful, self-hosted, drug-repurposing Agent. It eliminates fabricated citations by construction, abstains instead of inventing, measures and names the exact structural reason that novel repurposing is out of reach at 2-hop, scoping the fix. The result is a system whose trustworthiness is a property of its design and whose limitations provide a map for what is next. In a domain where a confident wrong answer is worse than an honest "I can't verify that," the agent has demonstrably shown not to give the wrong answer.

---

## References

Abdullahi, T., Gemou, I., Nayak, N. V., Murtaza, G., Bach, S. H., Eickhoff, C., & Singh, R. (2025). K-Paths: Reasoning over graph paths for drug repurposing and drug interaction prediction. In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (V.2)* (pp. 5–16).

Agrawal, A., Mackey, L., & Kalai, A. T. (2023). *Do language models know when they're hallucinating references?* arXiv:2305.18248.

Asgari, E., Montaña-Brown, N., Dubois, M., Khalil, S., Balloch, J., Au Yeung, J., & Pimenta, D. (2025). A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation. *NPJ Digital Medicine, 8*(1), 274.

Bonner, S., Barrett, I. P., Ye, C., Swiers, R., Engkvist, O., Hoyt, C. T., & Hamilton, W. L. (2022). Understanding the performance of knowledge graph embeddings in drug discovery. *Artificial Intelligence in the Life Sciences, 2*, 100036. arXiv:2105.10488.

Brown, A. S., & Patel, C. J. (2017). A standard database for drug repositioning. *Scientific Data, 4*, 170029. https://doi.org/10.1038/sdata.2017.29

Celebi, R., Uyar, H., Yasar, E., Gumus, O., Dikenelli, O., & Dumontier, M. (2019). Evaluation of knowledge graph embedding approaches for drug-drug interaction prediction in realistic settings. *BMC Bioinformatics, 20*(1), 726.

Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data, 10*(1), 67.

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems (NeurIPS) 30*.

Huang, K., Fu, T., Gao, W., Zhao, Y., Roohani, Y., Leskovec, J., … & Zitnik, M. (2024). A foundation model for clinician-centered drug repurposing. *Nature Medicine, 30*(9), 2635–2646.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys, 55*(12), 1–38.

Kamath, A., Jia, R., & Liang, P. (2020). Selective question answering under domain shift. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*. arXiv:2006.09462.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP)* (pp. 611–626). https://doi.org/10.1145/3600006.3613165

Lee, N., De Brouwer, E., Hajiramezanali, E., Park, C., & Scalia, G. (2025). RAG-enhanced collaborative LLM agents for drug discovery (CLADD). In *Proceedings of the AAAI Conference on Artificial Intelligence*.

Liu, F., Shareghi, E., Meng, Z., Basaldella, M., & Collier, N. (2021). Self-alignment pretraining for biomedical entity representations (SapBERT). In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)* (pp. 4228–4238). https://doi.org/10.18653/v1/2021.naacl-main.334

Rashkin, H., Nikolaev, V., Lamm, M., Aroyo, L., Collins, M., Das, D., Petrov, S., Tomar, G. S., Turc, I., & Reitter, D. (2023). Measuring attribution in natural language generation models. *Computational Linguistics, 49*(4), 777–840. https://doi.org/10.1162/coli_a_00486

Sun, Z., Deng, Z.-H., Nie, J.-Y., & Tang, J. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. In *International Conference on Learning Representations (ICLR)*.

Toutanova, K., & Chen, D. (2015). Observed versus latent features for knowledge base and text inference. In *Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality* (pp. 57–66).

Wang, Z. P., Li, X., Matsumoto, N., Venkatesan, M., Chang, J. H., Moran, J., Choi, H., Li, B., Meng, Y., Hernandez, M. E., & Moore, J. H. (2025). Drug repurposing for Alzheimer's disease using a graph-of-thoughts based large language model to infer drug–disease relationships in a comprehensive knowledge graph (ESCARGOT). *BioData Mining, 18*(1), 51.

Willard, B. T., & Louf, R. (2023). *Efficient guided generation for large language models.* arXiv:2307.09702.
