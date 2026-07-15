"""
TULU LEGAL — SCRIPT-VARIANT COMPREHENSION TEST (Conditions I & J)
=====================================================================
Extends tulu_comprehension_baseline.py with two new no-RAG, no-priming
conditions, using the SAME 20 sentences transliterated into Tamil and
Malayalam script instead of Kannada script.

WHY: Condition H (Kannada-script Tulu, no RAG) scored 80% via priming /
35-70% bare, comfortably above chance -- but Devadiga & Chopra (2026)
found Kannada-script Tulu is "nearly indistinguishable from Kannada" to
models trained heavily on Kannada. That leaves it ambiguous whether the
model is genuinely parsing TULU, or just reading it AS Kannada thanks to
heavy lexical/script overlap.

Tamil and Malayalam are both real, well-resourced Dravidian languages the
model has certainly seen plenty of -- but Tulu words transliterated into
Tamil/Malayalam script do NOT visually collapse into real Tamil/Malayalam
words the way they do into real Kannada words (different script, less
surface-level lexical overlap with Tulu specifically). If comprehension
holds up here too, that's real evidence for genuine cross-Dravidian
SEMANTIC transfer. If it collapses, that's evidence the Kannada-script
result was substantially a "read as Kannada" artifact.

IMPORTANT CAVEAT: the Tamil/Malayalam sentences here are MECHANICALLY
transliterated from the Kannada-script version via aksharamukha, NOT
independently written or reviewed by a native speaker of any of these
languages. This is a fast diagnostic pre-check, not a publication-ready
dataset -- flag this clearly if these numbers make it into the paper, and
ideally have a native Tulu speaker sanity-check a sample before relying on
this beyond a go/no-go signal for further investigation.

Run in the same folder as tulu_comprehension_baseline.py, with the new
tulu_legal_20_script_variants.jsonl file also present.
"""

import sys
import os
import csv
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tulu_comprehension_baseline import (
    load_tulu_queries,
    run_condition,
)

SCRIPT_VARIANTS_PATH = "tulu_legal_20_script_variants.jsonl"


def load_script_variants(path=SCRIPT_VARIANTS_PATH):
    """Load the Tamil/Malayalam transliterated versions."""
    if not os.path.exists(path):
        print(f"[Conditions I/J skipped] {path} not found.")
        return None, None

    tamil_map, malayalam_map = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            eid = item["id"]
            tamil_map[eid] = item["tulu_tamil"]
            malayalam_map[eid] = item["tulu_malayalam"]

    print(f"[Conditions I/J enabled] {len(tamil_map)} Tamil-script and "
          f"{len(malayalam_map)} Malayalam-script sentences loaded.")
    print("  NOTE: mechanically transliterated, NOT native-speaker reviewed.")
    return tamil_map, malayalam_map


def run_script_variant_baseline():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"tulu_comprehension_script_variants_{timestamp}.csv"

    print("=" * 65)
    print("TULU LEGAL — SCRIPT-VARIANT COMPREHENSION TEST")
    print("No priming. No RAG. Same 20 sentences, different scripts.")
    print()
    print("Condition I: Tamil-script Tulu only")
    print("  -> tests comprehension transfer outside Kannada-script")
    print("     assimilation specifically")
    print("Condition J: Malayalam-script Tulu only")
    print("  -> same test, second unrelated-script control")
    print()
    print("Key comparison vs Condition H (Kannada-script, bare): if I and J")
    print("hold up near H's accuracy, that's evidence for genuine cross-")
    print("Dravidian semantic transfer. If they collapse toward chance,")
    print("that's evidence H's result was substantially script-assimilation,")
    print("not real Tulu comprehension.")
    print("=" * 65)

    queries = load_tulu_queries()
    tamil_map, malayalam_map = load_script_variants()

    if tamil_map is None:
        return

    output_file = open(output_filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(output_file)
    writer.writerow([
        "condition", "id", "display_text", "english_gloss",
        "list_primary", "list_secondary",
        "pred_top1", "pred_top2", "pred_top3",
        "top1_confidence", "hits1", "hits3", "partial_credit",
        "top1_reasoning",
    ])
    output_file.flush()

    all_results = {}

    all_results["I_TamilScript"] = run_condition(
        "I_TamilScript",
        "Condition I — Tamil-script Tulu only (no RAG, no priming)",
        queries,
        get_display_text=lambda q: tamil_map[q["id"]],
        script_label="Tulu (Tamil script)",
        writer=writer, output_file=output_file,
    )

    all_results["J_MalayalamScript"] = run_condition(
        "J_MalayalamScript",
        "Condition J — Malayalam-script Tulu only (no RAG, no priming)",
        queries,
        get_display_text=lambda q: malayalam_map[q["id"]],
        script_label="Tulu (Malayalam script)",
        writer=writer, output_file=output_file,
    )

    output_file.close()

    print(f"\n{'='*65}")
    print("FINAL COMPARISON")
    print(f"{'='*65}")
    print(f"{'Condition':<35} {'Hits@1':>8} {'Hits@3':>8}")
    print("-" * 55)
    for name, r in all_results.items():
        print(f"{name:<35} {r['hits1']:>7.1f}% {r['hits3']:>7.1f}%")
    print()
    print("Reference — Condition H (Kannada script, bare, from prior run):")
    print("  H_KannadaOnly                       35.0%    70.0%")
    print()
    print(f"Results saved to: {output_filename}")
    print("=" * 65)


if __name__ == "__main__":
    run_script_variant_baseline()
