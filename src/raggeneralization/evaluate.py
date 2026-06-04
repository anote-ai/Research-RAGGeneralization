from __future__ import annotations
import re
from collections import Counter
from .core import TransferResult


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
