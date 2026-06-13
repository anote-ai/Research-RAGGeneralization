# Research Design Document: RAG Generalization

## Vision Statement

Prove that RAG systems **silently degrade** when deployed outside their evaluation domain, and provide the community with **RAGGen-Bench**: a dataset, diagnostic toolkit, and the **DGP** (Domain Gap Predictor) that lets practitioners forecast RAG performance before deployment — preventing the silent failures that erode trust in enterprise AI products.

---

## Problem Statement & Novelty

RAG systems are evaluated on a held-out split of their training domain, then deployed on production queries that may differ substantially. The community lacks:

1. **A systematic characterization of silent degradation**: RAG systems can drop 20+ pp in performance across domain shifts without any obvious signal.
2. **A domain gap predictor**: No model exists that forecasts RAG performance degradation given source and target domain descriptions.
3. **A domain adaptation benchmark**: No standard test of whether fine-tuning, domain-specific chunking, or corpus augmentation effectively closes the domain gap.
4. **Failure mode taxonomy**: The specific failure patterns under domain shift are uncharacterized.

### Novel Contributions

| Contribution | Description |
|---|---|
| **RAGGen-Bench** | 3,000 QA pairs across 10 domain pairs (source → target), testing generalization |
| **Domain Gap Score (DGS)** | Vocabulary overlap + embedding distribution distance between source and target corpora |
| **DGP model** | Regression model predicting NDCG@10 drop from DGS, trained on 40 domain pairs |
| **Silent degradation detector** | Online monitor that flags when query distribution has shifted from eval domain |
| **Adaptation comparison** | First systematic comparison of 4 adaptation strategies across domain types |

### DGS & DGP Definition

```
Domain Gap Score (DGS) = α × VocabDistance + β × EmbeddingDistance

where:
  VocabDistance = 1 - Jaccard(top_10K_terms(source), top_10K_terms(target))
  EmbeddingDistance = Wasserstein_distance(source_embeddings, target_embeddings)
  α, β learned from training data

DGP: NDCG_drop = f(DGS)  [linear regression baseline; neural model for production]
```

---

## Research Objectives

1. Quantify **silent degradation**: measure NDCG@10 drop across 10 source→target domain pairs.
2. Validate **DGP**: predict degradation from DGS before deployment, achieving R² > 0.75.
3. Compare **adaptation strategies**: fine-tuning vs. corpus augmentation vs. domain-specific chunking vs. prompt adaptation.
4. Build a **silent degradation detector** with <5% false positive rate on stable domains.
5. Identify which **RAG architectures** generalize best across domain shifts.

---

## Dataset Construction

### Domain Pairs (10 pairs, 300 QA each)

| Source Domain | Target Domain | DGS (estimated) |
|---|---|---|
| Wikipedia general | Scientific papers | High |
| News articles | Legal documents | High |
| Medical literature | Veterinary records | Medium |
| Software docs | Hardware manuals | Medium |
| Financial reports | Startup pitch decks | Medium |
| Academic papers | Industry whitepapers | Low |
| English Wikipedia | French Wikipedia (translated) | Medium |
| Formal documents | Social media posts | Very High |
| Product reviews | Customer support tickets | Low |
| Historical documents | Contemporary news | Medium |

### QA Pair Construction
```
For each domain pair:
1. Build source corpus + eval QA pairs (standard RAG evaluation)
2. Build target corpus covering the same topics in different domain style
3. Generate 300 QA pairs from target corpus (human-verified gold passages)
4. Compute DGS between source and target corpora
5. Evaluate all RAG systems on target QA pairs (no re-training) → measure degradation
```

---

## Systems Under Evaluation

| System | Architecture | Notes |
|---|---|---|
| BM25 | Sparse | Generalization baseline |
| DPR | Dense | Standard dense |
| ColBERT-v2 | Late interaction | Strong dense baseline |
| E5-large | Dense | General SOTA |
| SPLADE | Sparse learned | Hybrid |
| RAG-Fusion | Query expansion | Multi-query |
| AgenticRAG (from our work) | Agentic | Our cross-paper connection |

---

## Experimental Design

### Baseline Experiment (Experiment 0)
**Protocol**: Evaluate all systems on source domain (in-distribution). Establish NDCG@10 baseline.

**Expected result**: ColBERT-v2 ≈ 0.71, E5-large ≈ 0.69. These match published BEIR numbers, validating our evaluation pipeline.

---

### Experiment 1: Silent Degradation Characterization
**Hypothesis**: All RAG systems degrade by ≥15 pp NDCG@10 on high-DGS domain pairs, and this degradation is not predictable from any metric computed on source domain alone.

**Protocol**:
1. Evaluate all systems on all 10 target domains (no adaptation).
2. Compute NDCG@10 drop = source_score - target_score.
3. Test correlation: does source_score predict degradation? (Expected: r < 0.3)
4. Compute DGS for all domain pairs.
5. Test: DGS vs. degradation correlation (Expected: r > 0.75).

**Expected results**:

| Domain Pair | DGS | ColBERT Drop | E5 Drop |
|---|---|---|---|
| Wiki → Sci papers | 0.71 | −22 pp | −19 pp |
| News → Legal | 0.68 | −18 pp | −21 pp |
| Medical → Veterinary | 0.41 | −8 pp | −6 pp |
| Academic → Industry | 0.22 | −3 pp | −2 pp |
| Formal → Social media | 0.89 | −31 pp | −28 pp |

- DGS-degradation correlation: r ≈ 0.82 (strong predictor)
- Source score alone: r ≈ 0.18 (weak predictor — silent degradation confirmed)

---

### Experiment 2: Domain Gap Predictor (DGP) Validation
**Hypothesis**: DGP trained on 30 domain pairs achieves R² > 0.75 on 10 held-out pairs, enabling deployment-time degradation forecasting.

**Protocol**:
1. Collect 40 domain pairs total (10 from RAGGen-Bench + 30 from existing BEIR datasets).
2. Train DGP on 30 pairs, evaluate on 10.
3. Compute R², MAE for NDCG@10 drop prediction.
4. Compare: linear DGP vs. neural DGP vs. vocabulary-only baseline.

**Expected results**:
- Vocabulary-only baseline: R² ≈ 0.51
- Linear DGP (DGS features): R² ≈ 0.73
- Neural DGP (small MLP): R² ≈ 0.81
- Mean absolute error: 3.2 pp (neural DGP) — accurate enough for go/no-go deployment decisions

---

### Experiment 3: Adaptation Strategy Comparison
**Hypothesis**: Domain-specific corpus augmentation is the most cost-effective adaptation strategy, outperforming fine-tuning in 7/10 domain pairs.

**Protocol**:
1. Apply 4 adaptation strategies to each domain pair:
   - A: Fine-tune embedding model on target domain samples (100 examples)
   - B: Augment retrieval corpus with target domain documents
   - C: Domain-specific chunking (adapt chunk size/strategy to target document type)
   - D: Prompt adaptation (domain-specific system prompt for generator)
2. Measure NDCG@10 improvement per strategy.
3. Measure cost per strategy (compute + data collection).

**Expected results**:
- Corpus augmentation (B): avg +12 pp improvement, $0.50/domain cost
- Prompt adaptation (D): avg +6 pp improvement, $0.01/domain cost
- Fine-tuning (A): avg +14 pp improvement, $50/domain cost
- Domain chunking (C): avg +5 pp improvement, $0.10/domain cost
- Recommendation: corpus augmentation is 28× more cost-efficient than fine-tuning per pp improvement

---

### Experiment 4: Silent Degradation Detector
**Hypothesis**: A query distribution shift detector achieves <5% false positive rate while catching >80% of high-degradation deployment scenarios.

**Protocol**:
1. Train detector on query embedding distributions from source domains.
2. Evaluate on streams of queries from: (a) source domain (should not trigger), (b) target domains (should trigger for high-DGS pairs).
3. Measure false positive and true positive rates across DGS thresholds.

**Expected results**:
- At DGS threshold 0.40: TPR ≈ 0.84, FPR ≈ 0.04
- Detector triggers appropriate alert on 8/10 domain pairs where NDCG drop > 10 pp
- False positive rate on stable distributions: 3.2%

---

## Expected Results Summary

| Finding | Result |
|---|---|
| Silent degradation confirmed | All systems degrade ≥15 pp on high-DGS pairs |
| DGP accuracy | R² = 0.81, MAE = 3.2 pp |
| Best adaptation | Corpus augmentation: +12 pp at $0.50/domain |
| Detector performance | 84% TPR, 4% FPR at optimal threshold |
| Source score as predictor | r = 0.18 (useless for forecasting degradation) |

**Primary claim**: Domain Gap Score predicts RAG degradation with R² = 0.81, enabling practitioners to forecast production performance before deployment — a fundamentally new capability for responsible RAG deployment.

---

## Why This Matters

**For researchers**: RAGGen-Bench and DGP fill a critical gap in understanding RAG robustness — essential as RAG systems are deployed in increasingly diverse enterprise settings.

**For practitioners**: DGP enables deployment-time risk assessment; the silent degradation detector provides ongoing monitoring.

**For Anote products**: Every Anote RAG product can embed DGP as a deployment health check, directly improving reliability and customer trust.

---

## Implementation Plan

```
research-raggeneralization/
├── data/
│   ├── domain_pairs/     # 10 source-target corpus pairs
│   └── qa_pairs/         # 3,000 QA pairs with gold passages
├── dgs/
│   ├── vocab_distance.py
│   ├── embedding_distance.py
│   └── dgs_compute.py
├── dgp/
│   ├── train_dgp.py
│   └── neural_dgp.py
├── detector/
│   └── shift_detector.py
├── experiments/
│   ├── exp0_baseline.py
│   ├── exp1_degradation.py
│   ├── exp2_dgp_validation.py
│   ├── exp3_adaptation.py
│   └── exp4_detector.py
```

---

## Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Domain pair construction | 6 weeks | 10 pairs, 3,000 QA pairs |
| DGS computation pipeline | 2 weeks | DGS for all pairs |
| DGP training | 3 weeks | Trained DGP model |
| Experiments | 5 weeks | All results |
| Paper writing | 4 weeks | ACL/SIGIR submission |

**Target venue**: SIGIR 2027 or ACL 2026

---

## Open Questions & Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| 40 domain pairs may not be enough for DGP | Medium | Augment with synthetic domain pairs |
| DGS calibration across domain types | Medium | Validate on 5 held-out pairs |
| Adaptation cost estimates may vary | Low | Report compute costs explicitly |

---

## Related Issues

- Product integration: RAG deployment health monitoring
- Related work audit: BEIR, MTEB, domain adaptation literature
- Reproducibility: DGS computation pipeline
- Connection to RetrievalBench: DGS can be integrated with LA-NDCG
