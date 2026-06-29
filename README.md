# RAGGeneralization 

> **Research question:** Do RAG pipelines trained/tuned on US legal contracts generalize to EU legal contracts, and which pipeline components (chunking, embedding, reranking, metadata) reduce domain transfer degradation?

## Overview

RAGGeneralization benchmarks domain transfer for RAG pipelines using the [CUAD](https://www.atticusprojectai.org/cuad) contract understanding dataset as a proxy. It measures F1 on span extraction tasks and quantifies the transfer gap when moving between source and target legal domains.

The full research vision (RAGGen-Bench, the Domain Gap Score / DGP, a silent-degradation
detector, and an adaptation-strategy comparison across 10 domain pairs) is specified in
[`DESIGN_DOC.md`](DESIGN_DOC.md). **What's implemented today is a smaller proof-of-concept
slice of that vision** -- see the disclosure below and [`PAPER_DRAFT.md`](PAPER_DRAFT.md) /
[`BLOG.md`](BLOG.md) for an explicit, section-by-section accounting of what is real vs.
projected.

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

**Disclosure: these numbers are simulated, not measured.** They are produced by
`GeneralizationBench.simulate_transfer` (see `src/raggeneralization/core.py`), which draws
from a seeded `random.Random(hash(config_name))` combined with hand-picked
`_CONFIG_DEGRADATION` constants calibrated to *look like* plausible RAG domain-transfer
degradation. **No real CUAD dataset, no real embedding/retrieval models, and no real
F1 evaluation were run to produce this table.** The underlying contract corpus is 7
hand-written synthetic templates (`SAMPLE_CONTRACT_TEXTS` in `src/raggeneralization/data.py`),
not the actual 510-contract CUAD dataset. Treat this table as an illustration of the
intended shape of the result, not a research finding.

| Pipeline Config | Source F1 | Target F1 | Transfer Gap | Relative Drop |
|----------------|-----------|-----------|--------------|---------------|
| fixed_512\|bm25 | 0.72 | 0.57 | 0.15 | 20.8% |
| sentence\|dense | 0.71 | 0.59 | 0.12 | 16.9% |
| recursive\|dense\|cross-encoder | 0.73 | 0.64 | 0.09 | 12.3% |
| semantic\|dense\|cross-encoder\|meta | **0.74** | **0.68** | **0.06** | **8.1%** |

*(These specific numbers are the simulation's deterministic output for these configs, not
an empirical claim. Run `python scripts/run_demo.py` to reproduce them yourself.)*

## Generalization Findings

The bullet points below describe the *pattern the simulation was designed to produce*
(reranking and metadata reduce simulated degradation by construction, via the
`_CONFIG_DEGRADATION` multipliers), not an empirically observed finding:

- Semantic chunking + cross-encoder reranking reduces simulated transfer gap by ~60% vs BM25 baseline
- Metadata-aware retrieval (jurisdiction, document type) reduces simulated relative drop by construction
- These directional hypotheses are plausible given the retrieval literature, but have not yet been validated against real CUAD data or real embedding models

## Domain Gap Score (DGS)

`src/raggeneralization/dgs.py` implements a real, runnable version of the Domain Gap Score
from `DESIGN_DOC.md`: vocabulary-overlap (Jaccard) distance plus a dependency-free
token-distribution-distance proxy for the design doc's embedding/Wasserstein-distance term.
See `tests/test_dgs.py` for runnable tests, including a real (if small-scale) measurement
over the existing synthetic contract templates. This is the first concrete code for the
design doc's central predictive contribution; the DGP regression model itself, the
silent-degradation detector, and the full 10-domain-pair RAGGen-Bench are not yet implemented
(tracked as future work).

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/run_demo.py
pytest tests/ -v
```

## Target Venue

- **EMNLP 2026 NLLP Workshop** -- Natural Legal Language Processing

## Citation

```bibtex
@software{raggeneralization2026,
  title  = {RAGGeneralization: Domain Transfer Evaluation for RAG on Contract Data},
  author = {Anote AI},
  year   = {2026},
  url    = {https://github.com/anote-ai/research-raggeneralization}
}
```
