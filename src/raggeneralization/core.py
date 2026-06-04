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

        contract_types = ["NDA", "Software License", "Employment", "Service Agreement", "Lease"]
        docs: list[ContractDocument] = []
        for i in range(n):
            text = SAMPLE_CONTRACT_TEXTS[i % len(SAMPLE_CONTRACT_TEXTS)]
            docs.append(
                ContractDocument(
                    contract_id=f"contract_{i:04d}",
                    text=text,
                    contract_type=contract_types[i % len(contract_types)],
                    metadata={"index": i, "cuad_split": "train" if i < 8 else "test"},
                )
            )
        return docs

    def simulate_transfer(
        self,
        config: PipelineConfig,
        source_f1_boost: float = 0.0,
    ) -> TransferResult:
        """Return a realistic TransferResult for *config* with some degradation.

        Degradation is based on chunking strategy and flags (reranker, metadata
        both reduce the gap).
        """
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
