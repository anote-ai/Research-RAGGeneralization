from __future__ import annotations
import random
from .core import ContractDocument, CUADQuestion, TransferResult, PipelineConfig

# ---------------------------------------------------------------------------
# Richer CUAD-style contract texts covering more contract types
# ---------------------------------------------------------------------------

SAMPLE_CONTRACT_TEXTS: list[str] = [
    # 0 - NDA
    (
        "NON-DISCLOSURE AGREEMENT. "
        "This Non-Disclosure Agreement ('Agreement') is entered into as of January 1, 2025, "
        "between Acme Corp ('Disclosing Party') and Beta LLC ('Receiving Party'). "
        "The Receiving Party agrees to hold all Confidential Information in strict confidence "
        "and not to disclose it to any third party without prior written consent. "
        "Confidential Information means any non-public, proprietary information disclosed by "
        "the Disclosing Party in written, oral, or electronic form. "
        "This Agreement shall be governed by the laws of the State of Delaware. "
        "Either party may terminate this Agreement upon thirty (30) days written notice. "
        "Disputes arising under this Agreement shall be resolved by binding arbitration "
        "administered by the American Arbitration Association in Wilmington, Delaware."
    ),
    # 1 - Software License
    (
        "SOFTWARE LICENSE AGREEMENT. "
        "This Software License Agreement is effective as of March 15, 2025, between "
        "TechSoft Inc ('Licensor') and Enterprise Solutions Ltd ('Licensee'). "
        "Licensor grants Licensee a non-exclusive, non-transferable license to use the Software "
        "solely for internal business purposes. "
        "Licensee shall pay an annual license fee of $50,000 due on the first day of each year. "
        "Licensor's total liability shall not exceed the fees paid in the preceding twelve months. "
        "All intellectual property rights in the Software and any derivative works remain with Licensor. "
        "Licensee may not sublicense, sell, resell, transfer, assign, or otherwise commercially "
        "exploit or make available to any third party the Software. "
        "This Agreement shall be governed by the laws of the State of California."
    ),
    # 2 - Service Agreement
    (
        "SERVICE AGREEMENT. "
        "This Service Agreement ('Agreement') is made as of June 1, 2025, by and between "
        "Consulting Partners LLP ('Service Provider') and Global Industries Inc ('Client'). "
        "Service Provider shall deliver professional consulting services as described in "
        "Exhibit A attached hereto. "
        "Client shall pay Service Provider $200 per hour within 30 days of invoice receipt. "
        "This Agreement shall expire on May 31, 2026, unless renewed by mutual written agreement. "
        "This Agreement shall be construed under the laws of New York. "
        "Either party may terminate this Agreement for convenience upon sixty (60) days written notice. "
        "Service Provider shall indemnify and hold harmless Client from any third-party claims "
        "arising from Service Provider's gross negligence or wilful misconduct."
    ),
    # 3 - Employment Agreement
    (
        "EMPLOYMENT AGREEMENT. "
        "This Employment Agreement is entered into as of February 1, 2025, between "
        "Innovate Inc ('Employer') and Jane Doe ('Employee'). "
        "Employee shall serve as Vice President of Engineering commencing February 1, 2025. "
        "Employee's annual base salary shall be $320,000, payable bi-weekly. "
        "Employee is eligible for an annual performance bonus of up to 30% of base salary. "
        "Employee shall not, during employment and for 12 months thereafter, directly or "
        "indirectly compete with Employer's business in any jurisdiction where Employer operates. "
        "This Agreement is governed by the laws of the Commonwealth of Massachusetts. "
        "Either party may terminate employment at will upon four (4) weeks written notice."
    ),
    # 4 - Lease Agreement
    (
        "COMMERCIAL LEASE AGREEMENT. "
        "This Commercial Lease Agreement is entered into as of April 1, 2025, between "
        "Realty Holdings LLC ('Landlord') and StartUp Co ('Tenant'). "
        "Landlord agrees to lease to Tenant the premises located at 100 Main Street, "
        "San Francisco, California 94105 (the 'Premises'). "
        "The Lease Term shall commence April 1, 2025 and expire March 31, 2028. "
        "Tenant shall pay monthly rent of $15,000, due on the first day of each month. "
        "Tenant shall pay a security deposit of $45,000 upon execution of this Agreement. "
        "Tenant may not sublease or assign this Lease without Landlord's prior written consent. "
        "This Agreement shall be governed by the laws of the State of California."
    ),
    # 5 - Partnership Agreement
    (
        "GENERAL PARTNERSHIP AGREEMENT. "
        "This General Partnership Agreement is made as of September 1, 2025, among "
        "Alpha Ventures ('Partner A'), Beta Capital ('Partner B'), and Gamma Group ('Partner C'). "
        "The Partners agree to conduct business as 'ABG Partners' (the 'Partnership'). "
        "Profits and losses shall be allocated: 50% to Partner A, 30% to Partner B, "
        "and 20% to Partner C. "
        "Major decisions requiring unanimous consent include: admission of new partners, "
        "dissolution of the Partnership, and capital calls exceeding $1 million. "
        "Any Partner may withdraw upon ninety (90) days written notice to the other Partners. "
        "This Agreement is governed by the laws of the State of New York."
    ),
    # 6 - Franchise Agreement
    (
        "FRANCHISE AGREEMENT. "
        "This Franchise Agreement ('Agreement') is entered into as of July 1, 2025, between "
        "BurgerBrand Corp ('Franchisor') and Local Food LLC ('Franchisee'). "
        "Franchisor grants Franchisee the right to operate a BurgerBrand restaurant at "
        "250 Oak Avenue, Austin, Texas 78701 for a term of ten (10) years. "
        "Franchisee shall pay an initial franchise fee of $45,000 and an ongoing royalty "
        "of 6% of gross sales, payable monthly. "
        "Franchisee must adhere to Franchisor's Operations Manual and brand standards at all times. "
        "Upon expiration or termination, Franchisee shall immediately cease use of all "
        "Franchisor trademarks, trade dress, and proprietary systems. "
        "This Agreement is governed by the laws of the State of Texas."
    ),
]


def make_contract(i: int = 0) -> ContractDocument:
    """Return a synthetic ContractDocument at index *i*."""
    texts = SAMPLE_CONTRACT_TEXTS
    types = [
        "NDA",
        "Software License",
        "Service Agreement",
        "Employment",
        "Lease",
        "Partnership Agreement",
        "Franchise Agreement",
    ]
    return ContractDocument(
        contract_id=f"contract_{i:04d}",
        text=texts[i % len(texts)],
        contract_type=types[i % len(types)],
        metadata={"index": i, "jurisdiction": "US" if i % 2 == 0 else "EU"},
    )


def make_cuad_question(category: str = "Parties") -> CUADQuestion:
    """Return a realistic CUADQuestion for the given *category*."""
    questions: dict[str, tuple[str, list[str]]] = {
        "Parties": (
            "Who are the parties to this agreement?",
            ["Acme Corp", "Beta LLC"],
        ),
        "Effective Date": (
            "What is the effective date of this agreement?",
            ["January 1, 2025"],
        ),
        "Expiration Date": (
            "When does this agreement expire?",
            ["May 31, 2026"],
        ),
        "Governing Law": (
            "What law governs this agreement?",
            ["laws of the State of Delaware"],
        ),
        "Termination for Convenience": (
            "Can either party terminate for convenience, and if so on what notice?",
            ["thirty (30) days written notice", "sixty (60) days written notice"],
        ),
        "Payment Terms": (
            "What are the payment terms, including amounts and due dates?",
            ["$200 per hour within 30 days", "annual license fee of $50,000"],
        ),
        "Liability Cap": (
            "Is there a cap on liability, and if so what is it?",
            ["fees paid in the preceding twelve months"],
        ),
        "IP Ownership": (
            "Who owns the intellectual property created under this agreement?",
            ["All intellectual property rights in the Software remain with Licensor"],
        ),
        "Non-Compete": (
            "Does the agreement contain a non-compete clause, and if so what is its scope?",
            ["12 months", "directly or indirectly compete with Employer's business"],
        ),
        "Confidentiality": (
            "What confidentiality obligations are imposed on the parties?",
            ["hold all Confidential Information in strict confidence",
             "not to disclose it to any third party without prior written consent"],
        ),
        "Dispute Resolution": (
            "How are disputes resolved under this agreement?",
            ["binding arbitration", "American Arbitration Association"],
        ),
        "Indemnification": (
            "Who indemnifies whom, and under what circumstances?",
            ["Service Provider shall indemnify and hold harmless Client",
             "arising from Service Provider's gross negligence or wilful misconduct"],
        ),
    }
    question_text, spans = questions.get(category, (f"What is the {category}?", []))
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
