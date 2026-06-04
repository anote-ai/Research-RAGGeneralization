"""Evaluation metrics for RAG generalization experiments."""

from __future__ import annotations

from collections import Counter, defaultdict

from raggeneralization.core import TransferResult


def span_f1(predicted_spans: list[str], reference_spans: list[str]) -> float:
    """Token-level F1 over answer span sets.

    Each span is tokenised; total predicted and reference token bags are
    compared to compute precision, recall, and F1.
    """
    if not predicted_spans and not reference_spans:
        return 1.0
    if not predicted_spans or not reference_spans:
        return 0.0

    pred_tokens = Counter(_tokenise_all(predicted_spans))
    ref_tokens = Counter(_tokenise_all(reference_spans))

    common = sum((pred_tokens & ref_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall = common / sum(ref_tokens.values())
    return 2 * precision * recall / (precision + recall)


def transfer_degradation(source_f1: float, target_f1: float) -> float:
    """Relative performance drop from source to target domain.

    Returns a positive value when performance degrades (source > target).
    Negative values indicate the pipeline generalises better on the target.
    """
    if source_f1 == 0.0:
        return 0.0
    return (source_f1 - target_f1) / source_f1


def technique_generalization_report(results: list[TransferResult]) -> dict:
    """Group transfer results by pipeline config key and compute mean transfer gap.

    Returns:
        Dict mapping config_key (str) -> {mean_transfer_gap, mean_source_f1, mean_target_f1}.
    """
    grouped: dict[str, list[TransferResult]] = defaultdict(list)
    for result in results:
        key = _config_key(result)
        grouped[key].append(result)

    report = {}
    for key, group in grouped.items():
        n = len(group)
        report[key] = {
            "mean_transfer_gap": sum(r.transfer_gap for r in group) / n,
            "mean_source_f1": sum(r.source_f1 for r in group) / n,
            "mean_target_f1": sum(r.target_f1 for r in group) / n,
        }
    return report


def _tokenise_all(spans: list[str]) -> list[str]:
    """Flatten and tokenise a list of text spans."""
    tokens = []
    for span in spans:
        tokens.extend(span.lower().split())
    return tokens


def _config_key(result: TransferResult) -> str:
    cfg = result.config
    return (
        f"chunking={cfg.chunking}|embedding={cfg.embedding}"
        f"|reranker={cfg.reranker}|metadata={cfg.use_metadata}"
    )
