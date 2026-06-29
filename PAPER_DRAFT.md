# RAGGen-Bench: Characterizing and Predicting Silent Domain Degradation in RAG Systems

**Status: DRAFT SKELETON.** This is a structural skeleton for the eventual paper, not a submission-ready manuscript. Sections are populated with the argument structure and placeholders. Any numeric result below is one of two kinds:

- **(measured)** -- actually computed by running code in this repository, with a path to the script/test that produced it.
- **(projected, pending full experiment run)** -- the expected result stated in `DESIGN_DOC.md`, not yet verified by an actual experiment. These are hypotheses, not findings, and must not be cited as results until replaced.

Target venue (per DESIGN_DOC.md): SIGIR 2027 or ACL 2026.

---

## Abstract (draft)

Retrieval-augmented generation (RAG) systems are typically evaluated in-distribution, on held-out splits of their tuning domain. We hypothesize that this masks **silent degradation**: substantial performance drops when systems are deployed on out-of-distribution queries and corpora that were never represented in evaluation. We introduce (i) RAGGen-Bench, a benchmark of paired source/target domain corpora for measuring this effect, and (ii) the Domain Gap Score (DGS), a corpus-level statistic computable without labeled target-domain data, intended to predict degradation before deployment. *[Abstract numbers to be filled in once Experiments 0-4 below are actually run; currently no end-to-end run exists.]*

---

## 1. Introduction

- Motivating example: a RAG system tuned on US legal contracts (CUAD) deployed against EU contracts. (measured, small-scale): our synthetic CUAD demo (`scripts/run_demo.py`) shows simulated transfer gaps ranging from 0.06 to 0.15 F1 across four pipeline configurations -- illustrative of the *shape* of the problem, not a real-data finding (the underlying contract texts and F1 scores are synthetic/simulated, see Section 3).
- Gap in the literature: no standard benchmark isolates domain-shift-induced RAG degradation from other confounds (model choice, prompt design); no deployment-time predictor for that degradation exists (per DESIGN_DOC.md Problem Statement).
- Contributions claimed by the design doc (not all implemented yet -- status noted):
  1. RAGGen-Bench: 3,000 QA pairs across 10 domain pairs. **Status: not implemented.** Only a single source/target pair (legal-US / legal-EU) exists, and its corpus is 7 synthetic contract templates, not real CUAD documents.
  2. Domain Gap Score (DGS). **Status: implemented** in `src/raggeneralization/dgs.py` (added in this revision), computing vocabulary Jaccard distance and a token-distribution distance proxy. Not yet validated against real degradation measurements.
  3. DGP regression model. **Status: not implemented.**
  4. Silent degradation detector. **Status: not implemented.**
  5. Adaptation strategy comparison. **Status: not implemented** (no fine-tuning, augmentation, chunking, or prompt-adaptation experiment code exists beyond the chunking *label* used in the simulation).

---

## 2. Related Work

*(todo: BEIR, MTEB, domain adaptation for dense retrieval, distribution-shift detection literature -- design doc flags this as an open "related work audit" task, not yet done)*

---

## 3. RAGGen-Bench: Dataset and Setup

### 3.1 Current implementation

The repository currently implements a single domain pair: legal-US -> legal-EU, using `ContractDocument` objects in `src/raggeneralization/data.py`. The corpus is **7 hand-written synthetic contract texts** (NDA, software license, service agreement, employment, lease, partnership, franchise), not the real CUAD dataset (510 contracts, 13K+ annotations) referenced in the README and design doc. `GeneralizationBench.load_cuad_sample()` generates synthetic documents by cycling through these 7 templates and tagging them with a `jurisdiction` metadata field (`US` if index is even, `EU` if odd) -- this is a labeling convention for the simulation, not a real jurisdictional corpus split.

### 3.2 Planned dataset (per design doc, not yet built)

10 domain pairs x 300 QA pairs = 3,000 QA pairs, spanning Wikipedia->scientific papers, news->legal, medical->veterinary, software->hardware docs, financial->pitch decks, academic->industry, English->French Wikipedia, formal->social media, reviews->support tickets, historical->contemporary news.

---

## 4. Domain Gap Score (DGS)

### 4.1 Definition

As specified in DESIGN_DOC.md:

```
DGS = alpha * VocabDistance + beta * EmbeddingDistance
VocabDistance = 1 - Jaccard(top_10K_terms(source), top_10K_terms(target))
EmbeddingDistance = Wasserstein_distance(source_embeddings, target_embeddings)
```

### 4.2 Implementation (this revision)

`src/raggeneralization/dgs.py` implements:

- `vocab_distance(corpus_a, corpus_b, top_k=10_000)`: exact Jaccard distance over the top-k most frequent tokens in each corpus, matching the design-doc definition (no approximation needed at the scale of the current synthetic corpus).
- `token_distribution_distance(corpus_a, corpus_b)`: an embedding-free proxy for the design doc's `EmbeddingDistance` term, computed as the total variation distance between unigram frequency distributions over the shared vocabulary. This is a deliberate simplification: the design doc calls for Wasserstein distance over dense embeddings, which requires an embedding model dependency not currently in `requirements.txt`. The TV-distance proxy is offered as an interim, dependency-free signal; swapping in a real embedding-based Wasserstein distance once `sentence-transformers` (or similar) is added as a dependency is flagged as future work (see Section 4.3 below and the tracking issue).
- `domain_gap_score(corpus_a, corpus_b, alpha=0.5, beta=0.5)`: the weighted combination, with alpha/beta as user-supplied weights (the design doc specifies these should be "learned from training data" via the not-yet-built DGP; for now they default to an even 0.5/0.5 split).

### 4.3 Measured results (real, computed by tests/test_dgs.py)

Running `domain_gap_score` over the 7 synthetic CUAD-style contract templates in `src/raggeneralization/data.py`, grouped by the `jurisdiction` metadata tag used in the simulation (even index = "US", odd index = "EU"):

- This produces a **(measured)** DGS value for the synthetic templates, asserted in `tests/test_dgs.py::test_dgs_on_synthetic_contracts` to be a finite value in [0, 1] and to be strictly greater than the DGS of a corpus against itself (which must be 0 for vocab distance and 0 for token distribution distance, hence DGS = 0).
- This is **not** a finding about real US vs EU legal contract domain gap -- the synthetic templates differ mainly by contract type (NDA vs lease vs employment, etc.), not by genuine jurisdictional drafting style, so the resulting DGS reflects template diversity, not jurisdiction. We flag this explicitly to avoid the number being mistaken for a real result.

### 4.4 DGS validation against degradation (not yet done)

Design-doc hypothesis: DGS-degradation correlation r ~ 0.82 **(projected, pending full experiment run)**; source-score-alone correlation r ~ 0.18 **(projected, pending full experiment run)**. Validating this requires real degradation measurements across multiple real domain pairs, which do not yet exist (see Section 3.1).

---

## 5. Experiments

Status table, mapped to DESIGN_DOC.md's experiment list:

| Experiment | Design doc goal | Implementation status | Result status |
|---|---|---|---|
| Exp 0: Baseline | NDCG@10 on source domain, validate pipeline vs BEIR numbers | Not implemented (no NDCG/BEIR eval code; only simulated F1) | (projected, pending full experiment run) |
| Exp 1: Silent degradation characterization | Measure NDCG drop across 10 domain pairs, correlate with DGS | Not implemented (1 simulated pair only, no NDCG) | (projected, pending full experiment run) |
| Exp 2: DGP validation | Train/validate DGP regression, target R^2 > 0.75 | Not implemented (no DGP model code) | (projected, pending full experiment run): R^2 = 0.81 per design doc |
| Exp 3: Adaptation comparison | Compare 4 adaptation strategies | Not implemented | (projected, pending full experiment run): corpus augmentation ~28x more cost-efficient than fine-tuning per design doc |
| Exp 4: Silent degradation detector | <5% FPR, >80% TPR shift detector | Not implemented | (projected, pending full experiment run): 84% TPR / 4% FPR per design doc |

The only experiment with real, runnable code today is the CUAD pipeline-comparison demo (`scripts/run_demo.py`), and its transfer numbers are explicitly simulated (see `GeneralizationBench.simulate_transfer`, which draws from a seeded `random.Random(hash(config_name))` rather than evaluating a real retrieval pipeline).

---

## 6. Discussion / Threats to Validity

- The single biggest validity threat right now: **no real retrieval or generation model has been run.** All current "transfer" and "adaptation" numbers in the codebase are from `random.Random`-seeded simulations calibrated by hand-picked degradation constants (`GeneralizationBench._CONFIG_DEGRADATION`), not measured retrieval quality.
- DGS as currently implemented uses a TV-distance proxy instead of embedding-based Wasserstein distance; this may behave differently from the design doc's intended metric, especially for domain pairs with similar vocabulary but different semantic content.
- The CUAD corpus is currently 7 synthetic templates, not the real 510-contract CUAD dataset; conclusions from the demo should not be generalized.

---

## 7. Conclusion (draft)

*To be written once Experiments 0-2 have real measured results.*

---

## Appendix A: Mapping design-doc claims to current evidence

| Design doc claim | Evidence today |
|---|---|
| "All systems degrade >=15pp NDCG@10 on high-DGS pairs" | (projected, pending full experiment run) -- no NDCG measured |
| "DGS-degradation correlation r ~ 0.82" | (projected, pending full experiment run) |
| "DGP achieves R^2 = 0.81" | (projected, pending full experiment run) -- no DGP exists |
| "Corpus augmentation 28x more cost-efficient than fine-tuning" | (projected, pending full experiment run) -- no adaptation experiments exist |
| "Detector: 84% TPR, 4% FPR" | (projected, pending full experiment run) -- no detector exists |
| DGS is computable today | (measured) -- see Section 4.3, computed on synthetic CUAD templates only |
