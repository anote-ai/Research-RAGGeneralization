"""
TULU LEGAL — TRILINGUAL PRIMING-TIER COMPREHENSION TEST
============================================================
6 conditions: {Kannada, Tamil, Malayalam} script x {sparse, full} priming tier.
Each condition is structurally identical to Condition E (priming + NO RAG) --
just varying script and how much category-diagnostic legal vocabulary the
priming block includes.

SPARSE tier: general vocab + verbs + grammar only. NO legal nouns. Tests
genuine comprehension/structural-transfer, since none of these words directly
name or strongly imply a specific LIST category.

FULL tier: sparse tier + legal nouns (landlord, employer, money, land, house,
etc.) -- the original Condition E design. Tests glossary-assisted
classification, a legitimate but DIFFERENT claim than genuine transfer.
Every word in this tier was checked against the 20 test sentences and
confirmed to directly name or strongly imply the correct category for at
least one sentence -- see the leakiness table from our discussion.

Requires: tulu_priming_glossary_trilingual.json and
tulu_legal_20_script_variants.jsonl in the same folder, plus
tulu_legal_rag_v8.py for load_tulu_queries / load_kannada_script_tulu /
category_list_block / LIST_CATEGORIES.

Run with Ollama + llama3 available locally.
"""

import sys
import os
import csv
import json
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama
from tulu_legal_rag_v8 import (
    load_tulu_queries,
    load_kannada_script_tulu,
    category_list_block,
    LIST_CATEGORIES,
)

GLOSSARY_PATH = "tulu_priming_glossary_trilingual.json"
SCRIPT_VARIANTS_PATH = "tulu_legal_20_script_variants.jsonl"

FEW_SHOT_EXAMPLES = """Here are two example classifications:

Example 1:
Situation: "My landlord kept my deposit money after I moved out."
{"category": "Housing", "confidence": 0.85, "reasoning": "Landlord-tenant deposit dispute"}

Example 2:
Situation: "Someone broke my fence and refused to fix it."
{"category": "Torts/Individuals", "confidence": 0.80, "reasoning": "Property damage caused by another person"}
"""


def load_glossary():
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_script_variants():
    tamil_map, malayalam_map = {}, {}
    with open(SCRIPT_VARIANTS_PATH, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            tamil_map[item["id"]] = item["tulu_tamil"]
            malayalam_map[item["id"]] = item["tulu_malayalam"]
    return tamil_map, malayalam_map


def build_priming_block(glossary, script_key, tier, script_name):
    """
    script_key: 'kannada' | 'tamil' | 'malayalam' -- which field of each
        glossary entry to use.
    tier: 'sparse' | 'full'
    """
    words = [g for g in glossary if tier == "full" or g["tier"] == "safe"]

    general = [w for w in words if w["meaning"] in
               ("man", "she", "child", "mother/parent", "fish")]
    legal = [w for w in words if w["tier"] == "leaky"]
    verbs = [w for w in words if w["meaning"] in
             ("to go", "to do/make", "to give", "to know", "to want",
              "return/back", "to end/finish")]
    grammar = [w for w in words if w not in general + legal + verbs]

    def fmt(items):
        return ", ".join(f"{w[script_key]}={w['meaning']}" for w in items)

    lines = [f"Vocabulary guide for Tulu in {script_name} script:"]
    if general:
        lines.append(f"Nouns: {fmt(general)}")
    if legal:
        lines.append(f"Legal nouns: {fmt(legal)}")
    if verbs:
        lines.append(f"Verbs: {fmt(verbs)}")
    if grammar:
        lines.append(f"Grammar: {fmt(grammar)}")
    lines.append(
        "SOV word order — verb always at end. Suffix -g/-k = 'to/towards'. "
        "Suffix -n/-d = 'of/belonging to'. Quotative marks embedded speech."
    )
    return "\n".join(lines)


def build_prompt(display_text, script_label, priming_block):
    categories = category_list_block()
    situation = (
        f"SITUATION (Tulu written in {script_label} script): \"{display_text}\"\n"
        f"No translation available. Use the vocabulary guide above and any "
        f"words you recognize in the query to understand the situation."
    )
    sentence_instructions = (
        "Step 1: In one sentence, describe in plain English what problem "
        "this person is facing.\n"
        "Step 2: Identify the best-fitting LIST category and explain why, "
        "ruling out at least two alternatives.\n"
        "Step 3: Provide your TOP 3 predictions as JSON:\n"
        "```json\n{\n"
        '  "top1": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."},\n'
        '  "top2": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."},\n'
        '  "top3": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}\n'
        "}\n```\n"
        "CRITICAL: Every category MUST be exactly one of:\n"
        + "\n".join(f"  - {c}" for c in LIST_CATEGORIES)
        + "\nDo NOT invent new category names."
    )
    return (
        f"You are a legal AI assistant classifying everyday legal situations.\n\n"
        f"The legal categories are:\n{categories}\n\n"
        f"IMPORTANT: These categories are NOT evenly distributed. "
        f"Do not default to the same category repeatedly. "
        f"Actively weigh each situation against ALL categories.\n\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        f"{priming_block}\n\n"
        f"━━━ CLASSIFY THIS SITUATION ━━━\n\n"
        f"{situation}\n\n"
        f"{sentence_instructions}"
    )


def extract_top3(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return [data[k] for k in ("top1", "top2", "top3") if k in data]
        except json.JSONDecodeError:
            pass
    matches = re.findall(r'\{[^{}]*"category"[^{}]*\}', text, re.DOTALL)
    results = []
    for m in matches[:3]:
        try:
            results.append(json.loads(m))
        except json.JSONDecodeError:
            pass
    return results or [{"category": "ERROR", "confidence": 0.0, "reasoning": "parse error"}]


def validate_categories(top3):
    validated = []
    for pred in top3:
        cat = pred.get("category", "ERROR")
        if cat not in LIST_CATEGORIES:
            best = min(LIST_CATEGORIES,
                       key=lambda c: -sum(w in cat.lower() for w in c.lower().split("/")))
            pred = {**pred, "category": best}
        validated.append(pred)
    while len(validated) < 3:
        validated.append({"category": LIST_CATEGORIES[0], "confidence": 0.0, "reasoning": "padding"})
    return validated[:3]


def score_hits(top3_categories, primary, secondary):
    valid = {primary} | ({secondary} if secondary else set())
    hits1 = top3_categories[0] == primary
    hits3 = any(c in valid for c in top3_categories)
    partial = (not hits1) and hits3
    return hits1, hits3, partial


def call_model_with_retry(prompt, max_retries=3, delay_seconds=2):
    """
    Calls Ollama with retries for transient connection errors (e.g. Windows
    WinError 10053, or any other momentary drop). Only gives up after
    max_retries consecutive failures.
    """
    import time
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.generate(model="llama3", prompt=prompt,
                                        options={"temperature": 0.1, "repeat_penalty": 1.1})
            return response["response"].strip()
        except Exception as e:
            last_error = e
            print(f"    [retry {attempt}/{max_retries}] {type(e).__name__}: {e}")
            time.sleep(delay_seconds)
    raise last_error


def run_condition(condition_name, queries, get_display_text, script_label,
                   priming_block, writer, output_file):
    print(f"\n{'━'*65}\nRUNNING: {condition_name}\n{'━'*65}")
    hits1_count = hits3_count = 0
    for query in queries:
        display_text = get_display_text(query)
        prompt = build_prompt(display_text, script_label, priming_block)
        # Fallback is now a full 3-item list -- never causes an IndexError
        # downstream even if every retry fails.
        top3 = [{"category": "ERROR", "confidence": 0.0, "reasoning": "generation failed"}] * 3
        try:
            raw = call_model_with_retry(prompt)
            top3 = validate_categories(extract_top3(raw))
        except Exception as e:
            print(f"  GAVE UP on id {query['id']} after retries: {e}")

        cats = [p["category"] for p in top3]
        hits1, hits3, partial = score_hits(cats, query["list_primary"], query["list_secondary"])
        hits1_count += hits1
        hits3_count += hits3
        mark = "✓" if hits1 else ("~" if hits3 else "✗")
        print(f"  [{query['id']:>2}] {cats[0]:<22} {mark}  expected: {query['list_primary']}")

        writer.writerow([condition_name, query["id"], display_text, query["english_gloss"],
                          query["list_primary"], query["list_secondary"],
                          cats[0], cats[1], cats[2], f"{top3[0].get('confidence',0):.2f}",
                          hits1, hits3, partial, top3[0].get("reasoning", "")])
        output_file.flush()

    n = len(queries)
    print(f"\n  {condition_name}: Hits@1 {hits1_count/n*100:.1f}% ({hits1_count}/{n})  "
          f"Hits@3 {hits3_count/n*100:.1f}% ({hits3_count}/{n})")
    return {"hits1": hits1_count/n*100, "hits3": hits3_count/n*100, "n": n}


# If a previous run already completed some conditions cleanly, list them here
# to skip re-running them (copy their printed numbers into your notes first --
# this script does not merge with a prior partial CSV, it just skips re-doing
# the work in THIS run).
ALREADY_COMPLETED = {
    # "Kannada_sparse",   # 60.0% / 75.0% -- uncomment to skip
    # "Kannada_full",     # 70.0% / 90.0% -- uncomment to skip
}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"tulu_priming_tier_trilingual_{timestamp}.csv"

    glossary = load_glossary()
    queries = load_tulu_queries()
    kannada_map = load_kannada_script_tulu()
    tamil_map, malayalam_map = load_script_variants()

    scripts = [
        ("tamil", "Tamil", tamil_map),
        ("malayalam", "Malayalam", malayalam_map),
        ("kannada", "Kannada", kannada_map),
    ]

    output_file = open(output_filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(output_file)
    writer.writerow(["condition", "id", "display_text", "english_gloss",
                      "list_primary", "list_secondary", "pred_top1", "pred_top2",
                      "pred_top3", "top1_confidence", "hits1", "hits3",
                      "partial_credit", "top1_reasoning"])
    output_file.flush()

    all_results = {}
    for script_key, script_name, script_map in scripts:
        if script_map is None:
            print(f"[Skipped] {script_name} sentence map not available.")
            continue
        for tier in ("sparse", "full"):
            cond_name = f"{script_name}_{tier}"
            if cond_name in ALREADY_COMPLETED:
                print(f"[Skipped -- already completed] {cond_name}")
                continue
            priming_block = build_priming_block(glossary, script_key, tier, script_name)
            all_results[cond_name] = run_condition(
                cond_name, queries,
                get_display_text=lambda q, m=script_map: m[q["id"]],
                script_label=script_name,
                priming_block=priming_block,
                writer=writer, output_file=output_file,
            )

    output_file.close()

    print(f"\n{'='*65}\nFINAL COMPARISON\n{'='*65}")
    print(f"{'Condition':<25} {'Hits@1':>8} {'Hits@3':>8}")
    for name, r in all_results.items():
        print(f"{name:<25} {r['hits1']:>7.1f}% {r['hits3']:>7.1f}%")
    print(f"\nResults saved to: {output_filename}")


if __name__ == "__main__":
    main()