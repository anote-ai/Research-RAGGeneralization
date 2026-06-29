from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any

CUAD_CATEGORIES: list[str] = [
    "Parties",
    "Effective Date",
    "Expiration Date",
    "Governing Law",
    "Termination for Convenience",
    "Payment Terms",
    "Liability Cap",
    "IP Ownership",
    "Non-Compete",
    "Confidentiality",
    "Dispute Resolution",
    "Indemnification",
]


@dataclass
class ContractDocument:
    contract_id: str
    text: str
    contract_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CUADQuestion:
    question_id: str
    category: str
    question: str
    answer_spans: list[str] = field(default_factory=list)

    @property
    def has_answer(self) -> bool:
        """True if there is at least one non-empty answer span."""
        return any(s.strip() for s in self.answer_spans)


@dataclass
class PipelineConfig:
    chunking: str
    embedding: str
    reranker: str | None = None
    use_metadata: bool = False

    def name(self) -> str:
        parts = [self.chunking, self.embedding]
        if self.reranker:
            parts.append(self.reranker)
        if self.use_metadata:
            parts.append("meta")
        return "|".join(parts)


@dataclass
class TransferResult:
    source_domain: str
    target_domain: str
    config_name: str
    source_f1: float
    target_f1: float

    @property
    def transfer_gap(self) -> float:
        """Absolute F1 degradation when moving from source to target domain."""
        return self.source_f1 - self.target_f1

    @property
    def relative_drop(self) -> float:
        """Fractional F1 degradation relative to source performance."""
        if self.source_f1 <= 0:
            return 0.0
        return max(0.0, self.transfer_gap / self.source_f1)


@dataclass
class DomainAdaptationResult:
    """Captures how well a pipeline adapts from a source to a target domain.

    Attributes:
        config_name: Identifier for the retrieval pipeline configuration.
        source_domain: Domain used during pipeline training / tuning.
        target_domain: Unseen domain used for evaluation.
        source_ndcg: nDCG@10 on the source domain held-out set.
        target_ndcg: nDCG@10 on the target domain test set.
        adaptation_score: Composite adaptation score in [0, 1] defined as
            ``target_ndcg / source_ndcg`` (clamped to [0, 1]).  A score of
            1.0 means the pipeline transfers perfectly.
    """

    config_name: str
    source_domain: str
    target_domain: str
    source_ndcg: float
    target_ndcg: float

    @property
    def adaptation_score(self) -> float:
        """Ratio of target to source nDCG, clamped to [0, 1]."""
        if self.source_ndcg <= 0:
            return 0.0
        return min(1.0, self.target_ndcg / self.source_ndcg)

    @property
    def ndcg_gap(self) -> float:
        """Absolute nDCG drop from source to target."""
        return max(0.0, self.source_ndcg - self.target_ndcg)


# ---------------------------------------------------------------------------
# Cross-domain transfer matrix
# ---------------------------------------------------------------------------

def build_transfer_matrix(
    results: list[DomainAdaptationResult],
) -> dict[str, dict[str, float]]:
    """Build a domain-pair transfer matrix from a list of adaptation results.

    Returns a nested dict ``matrix[source_domain][target_domain]`` containing
    the mean :attr:`DomainAdaptationResult.adaptation_score` across all
    results sharing that (source, target) pair.  Diagonal entries (same
    source and target domain) are set to 1.0 by convention.

    Args:
        results: List of :class:`DomainAdaptationResult` objects.

    Returns:
        Nested dict representing the transfer matrix.
    """
    accumulator: dict[tuple[str, str], list[float]] = {}
    domains: set[str] = set()
    for r in results:
        key = (r.source_domain, r.target_domain)
        accumulator.setdefault(key, []).append(r.adaptation_score)
        domains.add(r.source_domain)
        domains.add(r.target_domain)

    matrix: dict[str, dict[str, float]] = {}
    for src in sorted(domains):
        matrix[src] = {}
        for tgt in sorted(domains):
            if src == tgt:
                matrix[src][tgt] = 1.0
            else:
                scores = accumulator.get((src, tgt), [])
                matrix[src][tgt] = sum(scores) / len(scores) if scores else 0.0
    return matrix


class GeneralizationBench:
    """Utilities for loading contract data and simulating domain transfer."""

    # Realistic degradation levels per config type (used in simulation)
    _CONFIG_DEGRADATION: dict[str, float] = {
        "fixed_512": 0.15,
        "sentence": 0.12,
        "recursive": 0.10,
        "semantic": 0.08,
    }

    def load_cuad_sample(self, n: int = 10) -> list[ContractDocument]:
        """Return *n* synthetic ContractDocuments mimicking CUAD structure."""
        from .data import SAMPLE_CONTRACT_TEXTS

        contract_types = [
            "NDA",
            "Software License",
            "Employment",
            "Service Agreement",
            "Lease",
            "Partnership Agreement",
            "Franchise Agreement",
        ]
        docs: list[ContractDocument] = []
        for i in range(n):
            text = SAMPLE_CONTRACT_TEXTS[i % len(SAMPLE_CONTRACT_TEXTS)]
            docs.append(
                ContractDocument(
                    contract_id=f"contract_{i:04d}",
                    text=text,
                    contract_type=contract_types[i % len(contract_types)],
                    metadata={
                        "index": i,
                        "cuad_split": "train" if i < 8 else "test",
                        "jurisdiction": "US" if i % 2 == 0 else "EU",
                    },
                )
            )
        return docs

    def simulate_transfer(
        self,
        config: PipelineConfig,
        source_f1_boost: float = 0.0,
    ) -> TransferResult:
        """Return a realistic TransferResult for *config* with some degradation."""
        rng = random.Random(hash(config.name()))
        base_degradation = self._CONFIG_DEGRADATION.get(config.chunking, 0.12)
        if config.reranker:
            base_degradation *= 0.75
        if config.use_metadata:
            base_degradation *= 0.85

        source_f1 = min(1.0, 0.70 + source_f1_boost + rng.uniform(-0.05, 0.05))
        noise = rng.uniform(-0.02, 0.02)
        target_f1 = max(0.0, source_f1 - base_degradation + noise)

        return TransferResult(
            source_domain="legal-US",
            target_domain="legal-EU",
            config_name=config.name(),
            source_f1=round(source_f1, 4),
            target_f1=round(target_f1, 4),
        )

    def simulate_adaptation(
        self,
        config: PipelineConfig,
        source_domain: str = "legal-US",
        target_domain: str = "legal-EU",
        source_ndcg_boost: float = 0.0,
    ) -> DomainAdaptationResult:
        """Simulate a :class:`DomainAdaptationResult` for the given config.

        Uses the same degradation logic as :meth:`simulate_transfer` but
        operates on nDCG scores rather than F1.
        """
        rng = random.Random(hash(config.name() + source_domain + target_domain))
        base_deg = self._CONFIG_DEGRADATION.get(config.chunking, 0.12)
        if config.reranker:
            base_deg *= 0.75
        if config.use_metadata:
            base_deg *= 0.85

        source_ndcg = min(1.0, 0.72 + source_ndcg_boost + rng.uniform(-0.05, 0.05))
        target_ndcg = max(0.0, source_ndcg - base_deg + rng.uniform(-0.02, 0.02))

        return DomainAdaptationResult(
            config_name=config.name(),
            source_domain=source_domain,
            target_domain=target_domain,
            source_ndcg=round(source_ndcg, 4),
            target_ndcg=round(target_ndcg, 4),
        )
