#!/usr/bin/env python3
"""Demo script: CUAD contract generalization benchmark."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raggeneralization.core import GeneralizationBench, PipelineConfig
from raggeneralization.evaluate import rank_by_generalization


def main() -> None:
    print("=== RAGGeneralization Demo ===")

    bench = GeneralizationBench()
    contracts = bench.load_cuad_sample(n=10)
    print(f"\nLoaded {len(contracts)} synthetic contracts.")
    for c in contracts[:3]:
        print(f"  {c.contract_id}: {c.contract_type} ({len(c.text)} chars)")

    configs = [
        PipelineConfig("fixed_512", "bm25"),
        PipelineConfig("sentence", "dense"),
        PipelineConfig("recursive", "dense", reranker="cross-encoder"),
        PipelineConfig("semantic", "dense", reranker="cross-encoder", use_metadata=True),
    ]

    results = [bench.simulate_transfer(cfg) for cfg in configs]

    print("\n--- Transfer Results ---")
    for r in results:
        print(
            f"  {r.config_name:<45}  "
            f"src_f1={r.source_f1:.4f}  "
            f"tgt_f1={r.target_f1:.4f}  "
            f"gap={r.transfer_gap:.4f}  "
            f"rel_drop={r.relative_drop:.2%}"
        )

    print("\n--- Generalization Ranking (best first) ---")
    ranking = rank_by_generalization(results)
    for row in ranking:
        print(
            f"  #{row['rank']}  {row['config_name']:<45}  "
            f"mean_gap={row['mean_transfer_gap']:.4f}"
        )


if __name__ == "__main__":
    main()
