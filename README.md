# Which RAG Techniques Generalize? Legal Contract Retrieval over CUAD

## Research Question

Does a high-performing RAG pipeline optimised for financial 10-K retrieval (FinanceBench) transfer to legal contract question answering (CUAD), and which pipeline components are responsible for domain-specific gains vs. general improvements?

## CUAD Dataset Overview

[CUAD](https://huggingface.co/datasets/cuad) (Contract Understanding Atticus Dataset) contains 510 commercial legal contracts annotated with 41 question categories covering critical legal clauses. We evaluate on a representative subset of 5 categories:

| Category | Description |
|----------|-------------|
| Parties | Names and roles of contracting entities |
| Effective Date | When the contract takes effect |
| Expiration Date | Contract end date or renewal conditions |
| Governing Law | Jurisdiction that governs the agreement |
| Termination for Convenience | Unilateral termination rights |

## Transfer Result Table (Template)

| Technique | Finance F1 | Legal F1 | Gap (pp) | Generalises? |
|-----------|------------|----------|----------|--------------|
| Baseline (BM25 + fixed-512) | — | — | — | — |
| + Sentence chunking | — | — | — | — |
| + Cross-encoder reranking | — | — | — | — |
| + LLM metadata injection | — | — | — | — |
| + Query expansion | — | — | — | — |
| Full pipeline | — | — | — | — |

## Generalization Findings

Preliminary hypotheses (to be validated empirically):

- **Chunking strategy** is expected to be domain-sensitive: sentence boundaries differ between financial prose and legal boilerplate.
- **Cross-encoder reranking** is expected to generalise well due to its domain-agnostic ranking signal.
- **LLM metadata annotation** may degrade on legal contracts if prompts were tuned for financial entities.
- **Query expansion** is expected to show mixed results depending on terminology overlap.

## Quickstart

```bash
git clone https://github.com/anote-ai/research-raggeneralization.git
cd research-raggeneralization
pip install -e ".[dev]"
pytest tests/ -v
```

```python
from raggeneralization.core import PipelineConfig, GeneralizationBench
from raggeneralization.evaluate import transfer_degradation

config = PipelineConfig(
    chunking="semantic",
    embedding="text-embedding-3-large",
    reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
    use_metadata=True,
)

bench = GeneralizationBench()
result = bench.evaluate_transfer(config)
relative_drop = transfer_degradation(result.source_f1, result.target_f1)
print(f"Transfer degradation: {relative_drop:.1%}")
```

## Citation

```bibtex
@misc{raggeneralization2024,
  title  = {Which RAG Techniques Generalize? Legal Contract Retrieval over CUAD},
  author = {Anote AI Research},
  year   = {2024},
  url    = {https://github.com/anote-ai/research-raggeneralization},
}

@article{cuad2021,
  title   = {CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author  = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal = {arXiv:2103.06268},
  year    = {2021},
}
```
