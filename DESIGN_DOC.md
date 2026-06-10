# RAGGeneralization — Research Design Document

## Goal

Quantify how much retrieval quality degrades when RAG systems are deployed outside their original domain — and provide practitioners with both a measurement framework and a set of domain adaptation techniques that can mitigate the degradation.

## Objective

1. Systematically measure retrieval quality degradation across 10 domain transfer scenarios
2. Characterize the "silent degradation" phenomenon: cases where the system appears to work but quality has dropped significantly
3. Benchmark 4 domain adaptation techniques and identify which works best under what conditions

## Background / Motivation

Every production RAG deployment eventually encounters domain shift. Practitioners assume retrieval quality is stable; in reality, NDCG@10 can drop 20–40% under significant domain shift while the system continues to return answers (silently wrong answers). BEIR captures this partially but doesn't characterize: the dimensions of shift that matter most, the silent degradation detection problem, or end-to-end RAG evaluation (retrieval + generation together).

## Experimental Design

### Baseline Experiment

**Measure NDCG@10 for E5-large and BM25 on BEIR's 18 source domains: in-domain fine-tuning vs. zero-shot**

- Metric: NDCG@10 in-domain vs. zero-shot; compute domain gap = (in-domain NDCG − zero-shot NDCG)
- Purpose: establish the range of domain gaps observed in BEIR; calibrate what "large" vs. "small" domain shift looks like
- Expected result: domain gaps range from ~2 points (similar domains) to ~15 points (very different domains)

### Test Experiment 1: Silent Degradation Detection

Deploy a RAG system tuned on domain A; gradually introduce domain B queries (10%, 30%, 50%, 100%). Test 3 detection methods: query distribution monitoring, retrieval confidence monitoring, answer consistency monitoring.

**Expected result:** NDCG@10 drops sharply at ~30% domain B queries; query distribution monitoring detects it earliest (at ~15%); answer consistency monitoring has 2–3x delay

### Test Experiment 2: Domain Adaptation Method Comparison

For 5 highest-gap domain transfers, evaluate: (1) zero-shot baseline, (2) query reformulation, (3) few-shot fine-tuning with 100 examples, (4) hybrid retrieval (domain embedding + target BM25).

**Expected result:** hybrid retrieval is best cost-effectiveness — provides 60% of fine-tuning NDCG improvement with zero labeled examples

### Test Experiment 3: Generalization Predictor

Use domain-level features (vocabulary overlap, document length ratio, query type similarity) to predict expected domain gap before deployment.

**Expected result:** 3-feature predictor achieves 0.75+ Spearman correlation with actual domain gap

## Expected Results

1. A cross-domain retrieval evaluation dataset spanning 10 domain transfer scenarios
2. Domain gap measurements for 5 leading retrieval systems
3. **Key finding:** "Silent retrieval degradation begins at 30% out-of-domain query mix — current monitoring detects it 2–3x too late"
4. Domain adaptation recommendation framework
5. A domain gap predictor: estimate expected degradation from metadata before deployment

## Why This Matters / Why People Would Care

- **RAG practitioners:** don't know when their retrieval is silently degrading — this paper tells them how to detect it and what to do
- **Embedding vendors** (Cohere, Voyage, OpenAI): their models' cross-domain behavior is a black box; systematic characterization helps them improve
- **ML researchers:** the silent degradation framing is novel and practically important
- **AI safety:** silently degrading retrieval is a safety issue in medical, legal, and financial contexts

## Timeline

| Month | Milestone |
|---|---|
| 1–2 | Dataset construction (10 domain pairs, relevance judgments) |
| 3 | Baseline measurements + silent degradation experiments |
| 4 | Domain adaptation method comparison |
| 5 | Domain gap predictor development |
| 6 | Submission to ACL 2026 or SIGIR 2027 |

## Related Issues

- Design doc GitHub issue: #19
- Target conferences: see issues labeled `conference-prep`
- Reproducibility package: see issues labeled `artifact-release`
