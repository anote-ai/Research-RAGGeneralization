from __future__ import annotations
import math
import re
from collections import Counter
from .core import TransferResult, DomainAdaptationResult


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def span_f1(predicted_spans: list[str], reference_spans: list[str]) -> float:
    """Token-level F1 across all spans (flattened).

    Both predicted_spans and reference_spans are lists of text strings.
    Tokens are accumulated across all spans before computing F1.
    """
    pred_tokens = [tok for span in predicted_spans for tok in _tokenize(span)]
    ref_tokens = [tok for span in reference_spans for tok in _tokenize(span)]
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = pred_counter & ref_counter
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def transfer_degradation(source_f1: float, target_f1: float) -> float:
    """Relative performance drop in [0, 1].  Clamped to non-negative."""
    if source_f1 <= 0:
        return 0.0
    return max(0.0, (source_f1 - target_f1) / source_f1)


def domain_adaptation_score(
    results: list[DomainAdaptationResult],
) -> dict[str, float]:
    """Compute the mean adaptation score per config across all domain pairs.

    The adaptation score for a single result is defined as
    ``min(1, target_ndcg / source_ndcg)`` (see
    :attr:`DomainAdaptationResult.adaptation_score`).  This function
    aggregates those scores across multiple source/target domain pairs.

    Args:
        results: List of :class:`DomainAdaptationResult` objects, potentially
            spanning multiple (source, target) domain pairs.

    Returns:
        Dict mapping ``config_name`` to mean adaptation score in [0, 1].
    """
    by_config: dict[str, list[float]] = {}
    for r in results:
        by_config.setdefault(r.config_name, []).append(r.adaptation_score)
    return {
        cfg: sum(scores) / len(scores)
        for cfg, scores in by_config.items()
    }


def cross_domain_ndcg(
    retrieved_lists: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int = 10,
) -> dict[str, float]:
    """Compute per-query nDCG@k for cross-domain retrieval evaluation.

    This metric mirrors standard IR nDCG but is designed for multi-domain
    benchmarks where each query comes from a potentially different domain.
    Binary relevance is assumed (1 if doc in qrels, else 0).

    Args:
        retrieved_lists: Dict mapping ``query_id`` to an ordered list of
            retrieved document IDs (most relevant first).
        qrels: Dict mapping ``query_id`` to a set of relevant document IDs.
        k: Cut-off rank for nDCG computation.

    Returns:
        Dict mapping ``query_id`` to its nDCG@k score, plus a ``'__mean__'``
        key containing the macro-average across all queries.
    """
    scores: dict[str, float] = {}
    for query_id, retrieved in retrieved_lists.items():
        relevant = qrels.get(query_id, set())
        scores[query_id] = _ndcg_at_k(retrieved, relevant, k)
    if scores:
        scores["__mean__"] = sum(scores.values()) / len(scores)
    else:
        scores["__mean__"] = 0.0
    return scores


def _ndcg_at_k(
    retrieved: list[str], relevant: set[str], k: int
) -> float:
    """Binary nDCG@k for a single query."""

    def dcg(ids: list[str]) -> float:
        return sum(
            (1.0 if ids[i] in relevant else 0.0) / math.log2(i + 2)
            for i in range(min(k, len(ids)))
        )

    actual_dcg = dcg(retrieved)
    ideal_ids = list(relevant) + [x for x in retrieved if x not in relevant]
    ideal_dcg = dcg(ideal_ids)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def technique_generalization_report(
    results: list[TransferResult],
) -> dict[str, dict[str, float]]:
    """Group results by config_name and compute aggregate transfer statistics.

    Returns:
        {config_name: {mean_source_f1, mean_target_f1, mean_transfer_gap, mean_relative_drop}}
    """
    by_config: dict[str, list[TransferResult]] = {}
    for r in results:
        by_config.setdefault(r.config_name, []).append(r)

    report: dict[str, dict[str, float]] = {}
    for config_name, rows in by_config.items():
        n = len(rows)
        report[config_name] = {
            "mean_source_f1": sum(r.source_f1 for r in rows) / n,
            "mean_target_f1": sum(r.target_f1 for r in rows) / n,
            "mean_transfer_gap": sum(r.transfer_gap for r in rows) / n,
            "mean_relative_drop": sum(r.relative_drop for r in rows) / n,
        }
    return report


def rank_by_generalization(results: list[TransferResult]) -> list[dict]:
    """Return configs ranked by ascending mean_transfer_gap (best generalizers first).

    Returns:
        List of dicts {config_name, mean_source_f1, mean_target_f1,
                       mean_transfer_gap, mean_relative_drop, rank}.
    """
    report = technique_generalization_report(results)
    ranked = sorted(report.items(), key=lambda x: x[1]["mean_transfer_gap"])
    output: list[dict] = []
    for rank, (config_name, stats) in enumerate(ranked, start=1):
        output.append({"config_name": config_name, "rank": rank, **stats})
    return output
