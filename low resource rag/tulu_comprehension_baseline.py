import ollama
import json
import csv
import re
import os
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# TULU LEGAL — BARE COMPREHENSION BASELINE
#
# Purpose: establish how much the model can classify purely from
# the raw sentence, with zero scaffolding — no priming, no RAG,
# no retrieved passages, no vocabulary guide, no grammar rules.
#
# Two conditions:
#   G. Romanized Tulu only (no RAG, no priming) — raw Latin-script
#      Tulu + category list only. Directly mirrors Condition B but
#      strips RAG. Answers: what can the model do with bare romanized
#      Tulu and no context at all? Is B's 10% from the model reading
#      the Tulu, or purely from lucky retrieved passages?
#   H. Kannada-script Tulu only (no RAG, no priming) — raw Kannada-
#      script Tulu + category list only. Directly mirrors Condition F
#      but strips RAG. Isolates the model's latent Kannada knowledge
#      with zero scaffolding.
#
# Key comparisons against v7 results:
#   G vs B (Romanized + RAG, 10%):      does RAG help romanized Tulu at all?
#   H vs F (Kannada + RAG, 40%):        does RAG help Kannada-script Tulu?
#   H vs E (Kannada + priming, 70%):    what does minimal priming add?
#   H vs D (Kannada + prime + RAG, 70%): what does the full pipeline add?
#   G vs H:                             how much does script alignment help
#                                        even without any scaffolding?
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TULU_DATASET_PATH = os.path.join(SCRIPT_DIR, "tulu_legal_20_list.jsonl")
TULU_KANNADA_CANDIDATE_PATHS = [
    os.path.join(SCRIPT_DIR, "tulu_legal_20_kannada.jsonl"),
    os.path.join(SCRIPT_DIR, "tulu_legal_20_kannada_DRAFT_needs_review.jsonl"),
]

LIST_CATEGORIES = [
    "Housing",
    "Consumer/Money/Debt",
    "Work/Employment",
    "Land/Property",
    "Torts/Individuals",
    "Courts/Legal-System",
    "IP/Identity",
    "Estates/Wills",
]

LIST_DEFINITIONS = {
    "Housing": (
        "landlord-tenant disputes, security deposits not returned, trespass into "
        "a home or dwelling, damage to rented property, eviction, someone entering "
        "your house without permission."
    ),
    "Consumer/Money/Debt": (
        "money owed but not paid, loans not repaid, overcharging beyond what was "
        "agreed, contract breaches involving payment, fraud where money was taken."
    ),
    "Work/Employment": (
        "wages not paid, wrongful dismissal without reason, employer stopping you "
        "from starting your own business, non-compete situations."
    ),
    "Land/Property": (
        "neighbor taking your land, someone occupying your land without permission "
        "and refusing to leave, boundary disputes, land stolen after inheritance."
    ),
    "Torts/Individuals": (
        "personal injury caused by another person, property damage caused by "
        "another person's negligence, someone breaking your property and not fixing it."
    ),
    "Courts/Legal-System": (
        "confusion about which court to go to, not knowing where to file a "
        "complaint, the other party being in a different location or jurisdiction, "
        "signing a document without understanding what it said."
    ),
    "IP/Identity": (
        "someone using your name without permission, someone selling things under "
        "your name or brand, misuse of personal or business identity."
    ),
    "Estates/Wills": (
        "inheritance disputes after a family member's death, family members taking "
        "property that was meant to be given to you."
    ),
}

FEW_SHOT_EXAMPLES = """Here are two example classifications:

Example 1:
Situation: "My landlord kept my deposit money after I moved out."
{"category": "Housing", "confidence": 0.85, "reasoning": "Landlord-tenant deposit dispute"}

Example 2:
Situation: "Someone broke my fence and refused to fix it."
{"category": "Torts/Individuals", "confidence": 0.80, "reasoning": "Property damage caused by another person"}
"""

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_tulu_queries(path=TULU_DATASET_PATH):
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            romanized = item.get("tulu", item.get("tulu_latin"))
            queries.append({
                "id":              item["id"],
                "tulu":            romanized,
                "english_gloss":   item["english_gloss"],
                "list_primary":    item.get("list_category", item.get("cuad_category")),
                "list_secondary":  item.get("list_category_secondary", ""),
            })
    print(f"Loaded {len(queries)} queries")
    return queries


def load_kannada_script_tulu(paths=TULU_KANNADA_CANDIDATE_PATHS):
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        print("[Condition H skipped] No Kannada-script file found.")
        return None
    entries, unreviewed = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            eid = item["id"]
            if item.get("tulu_kannada", "").strip():
                entries[eid] = item["tulu_kannada"].strip()
            elif item.get("tulu_kannada_DRAFT", "").strip():
                if item.get("reviewed_by_native_speaker") is True:
                    entries[eid] = item["tulu_kannada_DRAFT"].strip()
                else:
                    unreviewed.append(eid)
    missing = [i for i in range(1, 21) if i not in entries and i not in unreviewed]
    if unreviewed or missing:
        print(f"[Condition H skipped] Not ready — unreviewed: {sorted(unreviewed)}, missing: {sorted(missing)}")
        return None
    print(f"[Condition H enabled] {len(entries)} Kannada-script sentences loaded.")
    return entries

# ══════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ══════════════════════════════════════════════════════════════

def category_list_block():
    return "\n".join(
        f"- {c}: {LIST_DEFINITIONS[c]}" for c in LIST_CATEGORIES
    )


def build_prompt(display_text, script_label):
    """
    Bare prompt: category list + few-shot examples + situation.
    No priming, no retrieved passages, no vocabulary guide.
    """
    categories = category_list_block()
    situation = f"SITUATION (in {script_label}): \"{display_text}\""

    sentence_instructions = (
        "Step 1: In one sentence, describe in plain English what problem "
        "this person is facing.\n"
        "Step 2: Identify the best-fitting LIST category and explain why, "
        "ruling out at least two alternatives.\n"
        "Step 3: Provide your TOP 3 predictions as JSON:\n"
        "```json\n"
        "{\n"
        '  "top1": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."},\n'
        '  "top2": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."},\n'
        '  "top3": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}\n'
        "}\n"
        "```\n"
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
        f"━━━ CLASSIFY THIS SITUATION ━━━\n\n"
        f"{situation}\n\n"
        f"{sentence_instructions}"
    )

# ══════════════════════════════════════════════════════════════
# EXTRACTION + VALIDATION
# ══════════════════════════════════════════════════════════════

def extract_top3(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            results = []
            for key in ["top1", "top2", "top3"]:
                if key in data:
                    results.append({
                        "category":  data[key].get("category", "ERROR"),
                        "confidence": data[key].get("confidence", 0.0),
                        "reasoning":  data[key].get("reasoning", ""),
                    })
            if results:
                return results
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
            best = min(
                LIST_CATEGORIES,
                key=lambda c: -sum(w in cat.lower() for w in c.lower().split("/"))
            )
            pred = {**pred, "category": best,
                    "reasoning": f"[corrected from '{cat}'] " + pred.get("reasoning", "")}
        validated.append(pred)
    while len(validated) < 3:
        validated.append({"category": LIST_CATEGORIES[0], "confidence": 0.0,
                           "reasoning": "padding"})
    return validated[:3]


def score_hits(top3_categories, primary, secondary):
    valid_labels = {primary}
    if secondary:
        valid_labels.add(secondary)
    hits1 = top3_categories[0] == primary
    hits3 = any(c in valid_labels for c in top3_categories)
    partial = (not hits1) and hits3
    return hits1, hits3, partial

# ══════════════════════════════════════════════════════════════
# CONDITION RUNNER
# ══════════════════════════════════════════════════════════════

def run_condition(condition_name, label, queries, get_display_text,
                   script_label, writer, output_file):
    print(f"\n{'━'*65}")
    print(f"RUNNING: {label}")
    print(f"{'━'*65}")

    hits1_count = hits3_count = 0

    for query in queries:
        display_text = get_display_text(query)

        print(f"\n  [{query['id']}] {display_text[:60]}")
        print(f"  English: {query['english_gloss']}")
        print(f"  Expected: {query['list_primary']}"
              + (f" / {query['list_secondary']}" if query['list_secondary'] else ""))

        prompt = build_prompt(display_text, script_label)

        top3 = [{"category": "ERROR", "confidence": 0.0, "reasoning": "generation failed"}]
        try:
            response = ollama.generate(
                model="llama3",
                prompt=prompt,
                options={"temperature": 0.1, "repeat_penalty": 1.1},
            )
            raw = response["response"].strip()
            top3 = validate_categories(extract_top3(raw))
        except Exception as e:
            print(f"  ERROR: {e}")

        top3_cats = [p["category"] for p in top3]
        hits1, hits3, partial = score_hits(top3_cats, query["list_primary"], query["list_secondary"])

        if hits1:
            hits1_count += 1
        if hits3:
            hits3_count += 1

        h1 = "✓" if hits1 else ("~" if hits3 else "✗")
        print(f"  Top1: {top3_cats[0]} {h1}  Top2: {top3_cats[1]}  Top3: {top3_cats[2]}")

        writer.writerow([
            condition_name, query["id"], display_text, query["english_gloss"],
            query["list_primary"], query["list_secondary"],
            top3_cats[0], top3_cats[1], top3_cats[2],
            f"{top3[0]['confidence']:.2f}",
            hits1, hits3, partial,
            top3[0].get("reasoning", ""),
        ])
        output_file.flush()

    n = len(queries)
    print(f"\n  {label}")
    print(f"  Hits@1: {hits1_count/n*100:.1f}% ({hits1_count}/{n})")
    print(f"  Hits@3: {hits3_count/n*100:.1f}% ({hits3_count}/{n})")
    return {
        "hits1": hits1_count / n * 100,
        "hits3": hits3_count / n * 100,
        "hits1_n": hits1_count,
        "hits3_n": hits3_count,
        "total": n,
    }

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_baseline():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"tulu_comprehension_baseline_{timestamp}.csv"

    print("=" * 65)
    print("TULU LEGAL — BARE COMPREHENSION BASELINE")
    print("No priming. No RAG. Category list + few-shot examples only.")
    print()
    print("Condition G: Romanized Tulu only")
    print("  → mirrors Condition B but strips RAG")
    print("  → is B's 10% from reading Tulu, or from lucky retrieved passages?")
    print("Condition H: Kannada-script Tulu only")
    print("  → mirrors Condition F but strips RAG")
    print("  → pure latent Kannada knowledge, zero scaffolding")
    print()
    print("Key comparisons vs v7 results:")
    print("  G vs B (Romanized + RAG, 10%):   does RAG help romanized Tulu?")
    print("  H vs F (Kannada + RAG, 40%):     does RAG help Kannada script?")
    print("  H vs E (Kannada + prime, 70%):   what does priming add?")
    print("  G vs H:                          script alignment benefit, no scaffolding")
    print("=" * 65)

    queries = load_tulu_queries()
    kannada_script_map = load_kannada_script_tulu()

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

    # Condition G — Romanized Tulu, no RAG, no priming
    all_results["G_RomanizedOnly"] = run_condition(
        "G_RomanizedOnly",
        "Condition G — Romanized Tulu only (no RAG, no priming)",
        queries,
        get_display_text=lambda q: q["tulu"],
        script_label="Tulu (romanized)",
        writer=writer, output_file=output_file,
    )

    # Condition H — Kannada-script Tulu, no RAG, no priming
    if kannada_script_map:
        all_results["H_KannadaOnly"] = run_condition(
            "H_KannadaOnly",
            "Condition H — Kannada-script Tulu only (no RAG, no priming)",
            queries,
            get_display_text=lambda q: kannada_script_map[q["id"]],
            script_label="Tulu (Kannada script)",
            writer=writer, output_file=output_file,
        )
    else:
        print("\n[Condition H skipped] Kannada-script file not available.")

    output_file.close()

    print(f"\n{'='*65}")
    print("FINAL COMPARISON")
    print(f"{'='*65}")
    print(f"{'Condition':<35} {'Hits@1':>8} {'Hits@3':>8}")
    print(f"{'-'*55}")
    for name, res in all_results.items():
        print(f"{name:<35} {res['hits1']:>7.1f}% {res['hits3']:>7.1f}%")

    print()
    print("Reference numbers from v7 (nomic-embed-text):")
    print("  B (Romanized + RAG, no prime):   10.0% / 50.0%")
    print("  D (Kannada + prime + RAG):        70.0% / 95.0%")
    print("  E (Kannada + prime, no RAG):      70.0% / 95.0%")
    print("  F (Kannada + RAG, no prime):      40.0% / 70.0%")
    print()

    if "G_RomanizedOnly" in all_results:
        g = all_results["G_RomanizedOnly"]["hits1"]
        print(f"RAG value for romanized Tulu (B-G): {10.0-g:.1f} pp")
        print(f"  (what RAG adds to bare romanized Tulu)")

    if "H_KannadaOnly" in all_results:
        h = all_results["H_KannadaOnly"]["hits1"]
        print(f"RAG value for Kannada-script Tulu (F-H): {40.0-h:.1f} pp")
        print(f"  (what RAG adds to bare Kannada-script Tulu)")
        print(f"Priming value over bare Kannada (E-H): {70.0-h:.1f} pp")
        print(f"  (what the minimal 12-word Kannada priming adds)")
        print(f"Full pipeline value over bare Kannada (D-H): {70.0-h:.1f} pp")

    if "G_RomanizedOnly" in all_results and "H_KannadaOnly" in all_results:
        g = all_results["G_RomanizedOnly"]["hits1"]
        h = all_results["H_KannadaOnly"]["hits1"]
        print(f"Script alignment benefit, no scaffolding (H-G): {h-g:.1f} pp")
        print(f"  (how much Kannada script helps over romanized, with zero scaffolding)")

    print(f"\nResults saved to: {output_filename}")
    print("=" * 65)


if __name__ == "__main__":
    run_baseline()