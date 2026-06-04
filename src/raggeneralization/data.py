from __future__ import annotations
import random
from .core import ContractDocument, CUADQuestion, TransferResult, PipelineConfig

SAMPLE_CONTRACT_TEXTS: list[str] = [
    (
        "NON-DISCLOSURE AGREEMENT. "
        "This Non-Disclosure Agreement ('Agreement') is entered into as of January 1, 2025, "
        "between Acme Corp ('Disclosing Party') and Beta LLC ('Receiving Party'). "
        "The Receiving Party agrees to hold all Confidential Information in strict confidence "
        "and not to disclose it to any third party without prior written consent. "
        "This Agreement shall be governed by the laws of the State of Delaware. "
        "Either party may terminate this Agreement upon thirty (30) days written notice."
    ),
    (
        "SOFTWARE LICENSE AGREEMENT. "
        "This Software License Agreement is effective as of March 15, 2025, between "
        "TechSoft Inc ('Licensor') and Enterprise Solutions Ltd ('Licensee'). "
        "Licensor grants Licensee a non-exclusive, non-transferable license to use the Software "
        "solely for internal business purposes. "
        "Licensee shall pay an annual license fee of $50,000 due on the first day of each year. "
        "Licensor's liability shall not exceed the fees paid in the preceding twelve months. "
        "All intellectual property rights in the Software remain with Licensor."
    ),
    (
        "SERVICE AGREEMENT. "
        "This Service Agreement ('Agreement') is made as of June 1, 2025, by and between "
        "Consulting Partners LLP ('Service Provider') and Global Industries Inc ('Client'). "
        "Service Provider shall deliver professional consulting services as described in "
        "Exhibit A attached hereto. "
        "Client shall pay Service Provider $200 per hour within 30 days of invoice receipt. "
        "This Agreement shall expire on May 31, 2026, unless renewed by mutual written agreement. "
        "This Agreement shall be construed under the laws of New York."
    ),
]


def make_contract(i: int = 0) -> ContractDocument:
    """Return a synthetic ContractDocument at index *i*."""
    texts = SAMPLE_CONTRACT_TEXTS
    types = ["NDA", "Software License", "Service Agreement"]
    return ContractDocument(
        contract_id=f"contract_{i:04d}",
        text=texts[i % len(texts)],
        contract_type=types[i % len(types)],
        metadata={"index": i},
    )


def make_cuad_question(category: str = "Parties") -> CUADQuestion:
    """Return a synthetic CUADQuestion for the given *category*."""
    questions = {
        "Parties": ("Who are the parties to this agreement?", ["Acme Corp", "Beta LLC"]),
        "Effective Date": ("What is the effective date?", ["January 1, 2025"]),
        "Expiration Date": ("When does this agreement expire?", ["May 31, 2026"]),
        "Governing Law": ("What law governs this agreement?", ["laws of the State of Delaware"]),
        "Termination for Convenience": (
            "Can either party terminate for convenience?",
            ["thirty (30) days written notice"],
        ),
        "Payment Terms": ("What are the payment terms?", ["$200 per hour within 30 days"]),
        "Liability Cap": ("Is there a liability cap?", ["fees paid in the preceding twelve months"]),
        "IP Ownership": ("Who owns the IP?", ["All intellectual property rights remain with Licensor"]),
    }
    question_text, spans = questions.get(
        category, (f"What is the {category}?", [])
    )
    return CUADQuestion(
        question_id=f"q_{category.lower().replace(' ', '_')}",
        category=category,
        question=question_text,
        answer_spans=spans,
    )


def make_transfer_results(
    n_configs: int = 4, seed: int = 42
) -> list[TransferResult]:
    """Generate realistic TransferResults for *n_configs* pipeline configurations."""
    rng = random.Random(seed)
    configs = [
        PipelineConfig(chunking="fixed_512", embedding="bm25"),
        PipelineConfig(chunking="sentence", embedding="dense"),
        PipelineConfig(chunking="recursive", embedding="dense", reranker="cross-encoder"),
        PipelineConfig(
            chunking="semantic", embedding="dense", reranker="cross-encoder", use_metadata=True
        ),
    ][:n_configs]

    degradations = [0.15, 0.12, 0.09, 0.06]
    results: list[TransferResult] = []
    for cfg, base_deg in zip(configs, degradations):
        source_f1 = round(0.72 + rng.uniform(-0.04, 0.04), 4)
        noise = rng.uniform(-0.02, 0.02)
        target_f1 = round(max(0.0, source_f1 - base_deg + noise), 4)
        results.append(
            TransferResult(
                source_domain="legal-US",
                target_domain="legal-EU",
                config_name=cfg.name(),
                source_f1=source_f1,
                target_f1=target_f1,
            )
        )
    return results
