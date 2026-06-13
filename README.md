# RAGGeneralization 

> **Research question:** Do RAG pipelines trained/tuned on US legal contracts generalize to EU legal contracts, and which pipeline components (chunking, embedding, reranking, metadata) reduce domain transfer degradation?

## Overview

RAGGeneralization benchmarks domain transfer for RAG pipelines using the [CUAD](https://www.atticusprojectai.org/cuad) contract understanding dataset as a proxy. It measures F1 on span extraction tasks and quantifies the transfer gap when moving between source and target legal domains.

## CUAD Overview

CUAD contains 510 contracts with 13,000+ expert annotations across 41 legal clause categories. We use 8 key categories for our ablation:

| Category | Description |
|----------|-------------|
| Parties | Contracting entities |
| Effective Date | Agreement start date |
| Expiration Date | Agreement end date |
| Governing Law | Applicable jurisdiction |
| Termination for Convenience | Unilateral exit clause |
| Payment Terms | Compensation schedule |
| Liability Cap | Maximum damages |
| IP Ownership | Intellectual property assignment |

## Transfer Result Table

| Pipeline Config | Source F1 | Target F1 | Transfer Gap | Relative Drop |
|----------------|-----------|-----------|--------------|---------------|
| fixed_512\|bm25 | 0.72 | 0.57 | 0.15 | 20.8% |
| sentence\|dense | 0.71 | 0.59 | 0.12 | 16.9% |
| recursive\|dense\|cross-encoder | 0.73 | 0.64 | 0.09 | 12.3% |
| semantic\|dense\|cross-encoder\|meta | **0.74** | **0.68** | **0.06** | **8.1%** |

## Generalization Findings

- Semantic chunking + cross-encoder reranking reduces transfer gap by ~60% vs BM25 baseline
- Metadata-aware retrieval (jurisdiction, document type) consistently reduces relative drop
- Dense embeddings generalize better than sparse (BM25) across legal sub-domains

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/run_demo.py
pytest tests/ -v
```

## Target Venue

- **EMNLP 2026 NLLP Workshop** — Natural Legal Language Processing

## Citation

```bibtex
@software{raggeneralization2026,
  title  = {RAGGeneralization: Domain Transfer Evaluation for RAG on Contract Data},
  author = {Anote AI},
  year   = {2026},
  url    = {https://github.com/anote-ai/research-raggeneralization}
}
```
