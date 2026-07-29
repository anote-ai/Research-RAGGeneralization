"""
TULU LEGAL — MEGA PIPELINE (Full A-L Matrix x 3 Models)
================================================================================
Consolidates every scattered script from this project into one stable,
resumable pipeline. Runs all 19 condition-tier combinations (A, B, C_sparse,
C_full, D_sparse, D_full, E_sparse, E_full, F, G, H, I, J, K_sparse, K_full,
L_sparse, L_full, M_sparse, M_full) across three models, run in this order: Hex-1 (Ollama --
first, since it has the highest chance of crashing/collapsing, catching
issues early), Sarvam (API -- second), Llama3 (Ollama -- last, most
reliable, no reason to front-load it). 51 condition-runs total, ~1,020
individual calls.

ONE unified prompt-builder replaces the three different half-systems that
used to be scattered across v8, the trilingual priming script, and the bare
comprehension scripts -- one source of truth for script x priming-tier x RAG.

Every fix from this project's debugging history is folded in:
  - Sarvam: reasoning left at default, max_tokens=4000, content=None fallback,
    rate-limit pacing (1.1s min gap) + exponential backoff on 429s
  - Hex-1: think=False + /no_think appended + num_ctx=8192
  - Llama3: unchanged, always worked

RESUME IS FIRST-CLASS: point RESUME_FROM at any partial CSV (from any prior
run of this script, for any model) and it skips every (condition, id) pair
already completed, picking up exactly where it stopped. A crash never loses
more than the single row in flight.

SETUP REQUIRED:
  All data files in the same folder: kannada_legal_corpus_list.jsonl,
  tulu_legal_20_list.jsonl, tulu_legal_20_kannada_DRAFT_needs_review.jsonl,
  tulu_legal_20_script_variants.jsonl, tulu_priming_glossary_trilingual_v2.json
  tulu_legal_rag_v8.py in the same folder (shared engine functions).
  ollama pull hf.co/prithivMLmods/hex-1-f32-GGUF:Q4_K_M
  pip install sarvamai --break-system-packages
  Paste your Sarvam API key into SARVAM_API_KEY below.
"""

import sys
import os
import csv
import json
import re
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama
import numpy as np
from tulu_rag_engine import (
    LIST_CATEGORIES,
    category_list_block,
    format_retrieved_context,
    retrieve,
    embed_corpus,
    load_kannada_corpus,
    extract_top3,
    validate_categories,
    score_hits,
    USE_MEAN_CENTERING,
    mean_center,
    FEW_SHOT_EXAMPLES,
)

# v8's validate_categories() silently repairs a genuine API failure (the
# "ERROR" placeholder) by fuzzy-matching it against real category names --
# since "ERROR" matches none of them, all categories tie at score 0, and the
# tie-break always picks LIST_CATEGORIES[0] ("Housing"). This makes a total
# call failure look like a real, confident answer, and can even coincidentally
# register as a correct hit if the gold label happens to also be "Housing".
# This wrapper sends genuine failures to a sentinel that can NEVER match a
# real gold label, so a failed call always correctly scores as a miss.
PARSE_FAILURE_SENTINEL = "PARSE_FAILURE_NO_ANSWER"


def validate_categories_safe(top3):
    validated = []
    for pred in top3:
        cat = pred.get("category", "ERROR")
        if cat == "ERROR":
            pred = {**pred, "category": PARSE_FAILURE_SENTINEL}
        elif cat not in LIST_CATEGORIES:
            best = min(LIST_CATEGORIES,
                       key=lambda c: -sum(w in cat.lower() for w in c.lower().split("/")))
            pred = {**pred, "category": best}
        validated.append(pred)
    while len(validated) < 3:
        validated.append({"category": PARSE_FAILURE_SENTINEL, "confidence": 0.0,
                           "reasoning": "call failed -- no answer generated"})
    return validated[:3]

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "sk_pjuelwdd_YFcATIcYxOLcMF9EKHwNHRdm")
SARVAM_MODEL = "sarvam-30b"
HEX1_MODEL = "hf.co/prithivMLmods/hex-1-f32-GGUF:Q4_K_M"

# ── Folder layout: this script lives in pipeline/, with data/ and
#    outputs/ as sibling folders under the same project root ──
#      project-root/
#      ├── pipeline/   (this file + tulu_rag_engine.py)
#      ├── data/       (corpus, queries, glossary)
#      └── outputs/    (CSV results, created automatically if missing)
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

GLOSSARY_PATH = os.path.join(DATA_DIR, "tulu_priming_glossary_trilingual_v2.json")
CONSOLIDATED_QUERIES_PATH = os.path.join(DATA_DIR, "tulu_legal_60_consolidated.jsonl")
KANNADA_CORPUS_PATH = os.path.join(DATA_DIR, "kannada_legal_corpus_list.jsonl")

# Every file the pipeline needs, checked BEFORE the slow embedding step --
# so a missing/broken file is caught in seconds, not after a long wait.
REQUIRED_FILES = [
    KANNADA_CORPUS_PATH,
    CONSOLIDATED_QUERIES_PATH,
    GLOSSARY_PATH,
    os.path.join(PIPELINE_DIR, "tulu_rag_engine.py"),
]

# Point each of these at a partial CSV from a previous run of THIS script to
# resume that model. Leave None to run that model fully fresh.
# Point each of these at a partial CSV from a previous run to resume that
# model -- CSVs now live in outputs/, e.g. "../outputs/tulu_mega_sarvam_....csv"
# (or an absolute path). Leave None to run that model fully fresh.
RESUME_FROM = {
    "llama3": None,
    "sarvam": None,
    "hex1": None,
}

SARVAM_MIN_GAP_SECONDS = 1.1  # 60 req/min limit -> 1/sec; small margin added

# Run in 3 batches of 20 sentences each, all 19 conditions x 3 models per
# batch, rather than all 60 at once -- if a long run gets interrupted,
# whichever batches already finished are safe and independently usable.
BATCHES = [(1, 20), (21, 40), (41, 60)]


# ══════════════════════════════════════════════════════════════
# CONDITION TABLE — single source of truth, 19 unique runs
# ══════════════════════════════════════════════════════════════
# (condition_id, script, priming_tier, use_rag)
#   script: "english" | "romanized" | "kannada" | "tamil" | "malayalam"
#   priming_tier: None | "sparse" | "full"

CONDITIONS = [
    ("A",        "english",    None,     True),
    ("B",        "romanized",  None,     True),
    ("C_sparse", "romanized",  "sparse", True),
    ("C_full",   "romanized",  "full",   True),
    ("D_sparse", "kannada",    "sparse", True),
    ("D_full",   "kannada",    "full",   True),
    ("E_sparse", "kannada",    "sparse", False),
    ("E_full",   "kannada",    "full",   False),
    ("F",        "kannada",    None,     True),
    ("G",        "romanized",  None,     False),
    ("H",        "kannada",    None,     False),
    ("I",        "tamil",      None,     False),
    ("J",        "malayalam",  None,     False),
    ("K_sparse", "tamil",      "sparse", False),
    ("K_full",   "tamil",      "full",   False),
    ("L_sparse", "malayalam",  "sparse", False),
    ("L_full",   "malayalam",  "full",   False),
    ("M_sparse", "romanized",  "sparse", False),  # new: completes the D/E,
    ("M_full",   "romanized",  "full",   False),  # K, L pattern for Romanized
]

SCRIPT_DISPLAY_NAME = {
    "english": "English", "romanized": "Romanized Tulu", "kannada": "Kannada",
    "tamil": "Tamil", "malayalam": "Malayalam",
}


# ══════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECK — runs BEFORE embedding, so a broken/missing file
# is caught in seconds, not after a long wait for embeddings to finish.
# ══════════════════════════════════════════════════════════════

def preflight_check():
    print("=" * 70)
    print("PRE-FLIGHT CHECK — verifying all required files before embedding")
    print("=" * 70)
    ok = True

    for path in REQUIRED_FILES:
        exists = os.path.exists(path)
        print(f"  [{'OK' if exists else 'MISSING'}] {path}")
        if not exists:
            ok = False

    if not ok:
        print("\nABORTING: one or more required files are missing. Fix the above "
              "before running -- no embedding has started, no time wasted.")
        return False

    # Structural check on the consolidated queries file specifically --
    # this is the file that caused the last crash, so check it thoroughly.
    print(f"\n  Checking {CONSOLIDATED_QUERIES_PATH} structure...")
    required_fields = ["english_gloss", "list_category", "tulu_romanized",
                        "tulu_kannada", "tulu_tamil", "tulu_malayalam"]
    try:
        with open(CONSOLIDATED_QUERIES_PATH, encoding="utf-8") as f:
            rows = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}
    except Exception as e:
        print(f"  [MISSING/BROKEN] Could not parse {CONSOLIDATED_QUERIES_PATH}: {e}")
        return False

    problems = []
    if not rows:
        problems.append("consolidated file contains zero rows")
    else:
        expected_ids = range(1, max(rows.keys()) + 1)  # dynamic -- works for 20, 60, or any future size
        for i in expected_ids:
            if i not in rows:
                problems.append(f"id {i}: entire row missing (gap in id sequence)")
                continue
            empty = [f for f in required_fields if not rows[i].get(f, "").strip()
                      if f != "list_category_secondary"]
            if empty:
                problems.append(f"id {i}: missing fields {empty}")

    if problems:
        print(f"  [FAIL] {len(problems)} problem(s) found:")
        for p in problems:
            print(f"    - {p}")
        print("\nABORTING before embedding -- fix the consolidated file first.")
        return False

    print(f"  [OK] All {len(rows)} entries present with every required field populated.")
    print("\nPre-flight check passed. Proceeding to embedding...\n")
    return True


# ══════════════════════════════════════════════════════════════
# DATA LOADING — one consolidated file, one source of truth
# ══════════════════════════════════════════════════════════════

def load_glossary():
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_consolidated_queries():
    """
    Returns (queries, kannada_map, tamil_map, malayalam_map) all from ONE
    file, eliminating the multi-file-drift risk that caused the earlier
    crash (three separate files that could silently fall out of sync).
    """
    queries = []
    kannada_map, tamil_map, malayalam_map = {}, {}, {}
    with open(CONSOLIDATED_QUERIES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item["id"]
            queries.append({
                "id": qid,
                "tulu": item["tulu_romanized"],
                "english_gloss": item["english_gloss"],
                "list_primary": item["list_category"],
                "list_secondary": item.get("list_category_secondary", ""),
            })
            kannada_map[qid] = item["tulu_kannada"]
            tamil_map[qid] = item["tulu_tamil"]
            malayalam_map[qid] = item["tulu_malayalam"]
    print(f"Loaded {len(queries)} queries from single consolidated source "
          f"({CONSOLIDATED_QUERIES_PATH}).")
    return queries, kannada_map, tamil_map, malayalam_map


def get_display_text_fn(script, kannada_map, tamil_map, malayalam_map):
    """Returns a function query -> display text for the given script."""
    if script == "english":
        return lambda q: q["english_gloss"]
    elif script == "romanized":
        return lambda q: q["tulu"]
    elif script == "kannada":
        return lambda q: kannada_map[q["id"]]
    elif script == "tamil":
        return lambda q: tamil_map[q["id"]]
    elif script == "malayalam":
        return lambda q: malayalam_map[q["id"]]
    raise ValueError(f"Unknown script: {script}")


# ══════════════════════════════════════════════════════════════
# UNIFIED PRIMING-BLOCK BUILDER
# (generalizes tulu_priming_tier_trilingual.py's version to also
#  support "romanized" as a script_key -- works automatically now
#  that every glossary entry has a "romanized" field)
# ══════════════════════════════════════════════════════════════

def build_priming_block(glossary, script_key, tier):
    words = [g for g in glossary if tier == "full" or g["tier"] == "safe"]

    general = [w for w in words if w["meaning"] in
               ("man", "she", "child", "mother", "father", "fish")]
    legal = [w for w in words if w["tier"] == "leaky"]
    verbs = [w for w in words if w["meaning"] in
             ("to go", "to do/make", "to give", "to know", "to want",
              "return/back", "to end/finish")]
    grammar = [w for w in words if w not in general + legal + verbs]

    def fmt(items):
        return ", ".join(f"{w[script_key]}={w['meaning']}" for w in items)

    script_name = SCRIPT_DISPLAY_NAME[script_key if script_key != "romanized" else "romanized"] \
        if script_key != "romanized" else "Romanized Tulu"

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


# ══════════════════════════════════════════════════════════════
# UNIFIED PROMPT BUILDER — one source of truth for all 19 conditions
# ══════════════════════════════════════════════════════════════

def build_situation_block(script, display_text, has_priming):
    if script == "english":
        return f'SITUATION (in English): "{display_text}"'

    script_label = SCRIPT_DISPLAY_NAME[script]
    base = f'SITUATION (Tulu written in {script_label} script): "{display_text}"\n' \
        if script != "romanized" else f'SITUATION (Tulu, romanized): "{display_text}"\n'

    if has_priming:
        tail = ("No translation available. Use the vocabulary guide above and "
                "any words you recognize in the query to understand the situation.")
    else:
        tail = "No translation available. Use any patterns you can detect in the text."
    return base + tail


def build_unified_prompt(script, display_text, retrieved, priming_block, use_rag):
    static_prefix = (
        "You are a legal AI assistant helping classify everyday legal situations "
        "described by lay speakers into legal issue categories.\n\n"
        "The legal categories are:\n"
        f"{category_list_block()}\n\n"
        "IMPORTANT: These categories describe EVERYDAY PROBLEMS, not commercial "
        "contract clauses. Classify based on what the PERSON'S SITUATION is about, "
        "NOT based on legal terminology in retrieved passages.\n\n"
        "Do not default to the same category repeatedly. Weigh each situation "
        "against ALL categories before deciding.\n\n"
        f"{FEW_SHOT_EXAMPLES}"
    )
    if priming_block:
        static_prefix += f"\n{priming_block}\n\n"

    situation = build_situation_block(script, display_text, bool(priming_block))

    sentence_instructions = (
        "Step 1: Re-read the SITUATION above. In one sentence, describe in plain "
        "English what problem this person is facing — ignore the passages for now.\n"
        "Step 2: Based on YOUR description of the situation (not the passages), "
        "identify the best-fitting LIST category and explain why. Rule out at least "
        "two other categories explicitly.\n"
        "Step 3: Provide your TOP 3 category predictions in order, as JSON:\n"
        "```json\n{\n"
        '  "top1": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."},\n'
        '  "top2": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."},\n'
        '  "top3": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}\n'
        "}\n```\n"
        "CRITICAL: Every category value MUST be exactly one of:\n"
        + "\n".join(f"  - {c}" for c in LIST_CATEGORIES)
        + "\nDo NOT invent new category names or use terms from the passages."
    )

    if not use_rag or not retrieved:
        return (
            f"{static_prefix}"
            f"━━━ CLASSIFY THIS SITUATION ━━━\n\n{situation}\n\n"
            f"No retrieved passages are provided for this classification.\n"
            f"Classify based solely on the situation description and the "
            f"category definitions above.\n\n"
            f"REMINDER — classify this situation:\n{situation}\n\n"
            f"{sentence_instructions}"
        )

    ctx = format_retrieved_context(retrieved)
    return (
        f"{static_prefix}"
        f"━━━ CLASSIFY THIS SITUATION ━━━\n\n{situation}\n\n"
        f"The passages below are BACKGROUND LAW ONLY. Do NOT classify what the "
        f"passage is about — classify what the PERSON'S SITUATION is about.\n"
        f"Retrieved passages from the Kannada court corpus:\n{ctx}\n\n"
        f"REMINDER — classify this situation:\n{situation}\n\n"
        f"{sentence_instructions}"
    )


# ══════════════════════════════════════════════════════════════
# RESUME SYSTEM
# ══════════════════════════════════════════════════════════════

def load_completed(csv_path):
    """
    Returns {(condition, id): row_dict} from a prior (possibly partial) run.
    Rows that are actually failed calls masquerading as answers (confidence
    0.00, parse-error reasoning -- see PARSE_FAILURE_SENTINEL above) are
    deliberately EXCLUDED here, so a resume run retries them for real
    instead of permanently locking in a corrupted placeholder.
    """
    completed = {}
    skipped_corrupted = 0
    if not csv_path or not os.path.exists(csv_path):
        if csv_path:
            print(f"  [resume] WARNING: {csv_path} not found -- running fresh.")
        return completed
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            is_corrupted = (
                row.get("pred_top1") in ("Housing", "ERROR", PARSE_FAILURE_SENTINEL)
                and row.get("top1_confidence") in ("0.00", "0")
                and "parse error" in row.get("top1_reasoning", "").lower()
            ) or row.get("pred_top1") == PARSE_FAILURE_SENTINEL
            if is_corrupted:
                skipped_corrupted += 1
                continue
            completed[(row["condition"], row["id"])] = row
    print(f"  [resume] Loaded {len(completed)} already-completed rows from {csv_path}")
    if skipped_corrupted:
        print(f"  [resume] Excluded {skipped_corrupted} corrupted parse-error rows "
              f"-- these will be retried, not skipped.")
    return completed


# ══════════════════════════════════════════════════════════════
# MODEL CALL ADAPTERS — every fix from this project folded in
# ══════════════════════════════════════════════════════════════

def call_llama3(prompt, max_retries=3, delay_seconds=2, temperature=0.1):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.generate(model="llama3", prompt=prompt,
                                        options={"temperature": temperature, "repeat_penalty": 1.1})
            content = response["response"].strip()
            if not content:
                raise ValueError("Llama3 returned empty response")
            return content
        except Exception as e:
            last_error = e
            print(f"    [retry {attempt}/{max_retries}] {type(e).__name__}: {e}")
            time.sleep(delay_seconds)
    raise last_error


def call_hex1(prompt, max_retries=3, delay_seconds=3, temperature=0.1):
    prompt_with_nothink = f"{prompt}\n\n/no_think"
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                response = ollama.generate(
                    model=HEX1_MODEL, prompt=prompt_with_nothink, think=False,
                    options={"temperature": temperature, "repeat_penalty": 1.1, "num_ctx": 8192},
                )
            except TypeError:
                response = ollama.generate(
                    model=HEX1_MODEL, prompt=prompt_with_nothink,
                    options={"temperature": temperature, "repeat_penalty": 1.1, "num_ctx": 8192},
                )
            content = response["response"].strip()
            if not content:
                raise ValueError("Hex-1 returned empty response")
            return content
        except Exception as e:
            last_error = e
            print(f"    [retry {attempt}/{max_retries}] {type(e).__name__}: {e}")
            time.sleep(delay_seconds)
    raise last_error


_sarvam_last_call_time = [0.0]


def _sarvam_pace():
    elapsed = time.time() - _sarvam_last_call_time[0]
    if elapsed < SARVAM_MIN_GAP_SECONDS:
        time.sleep(SARVAM_MIN_GAP_SECONDS - elapsed)
    _sarvam_last_call_time[0] = time.time()


def get_sarvam_client():
    from sarvamai import SarvamAI
    # SDK defaults to a 60s timeout -- reasoning mode + max_tokens=4000 can
    # genuinely take longer than that, causing ReadTimeouts that exhaust all
    # retries. 120s gives real responses room to complete.
    return SarvamAI(api_subscription_key=SARVAM_API_KEY, timeout=180.0)


def call_sarvam(client, prompt, max_retries=7, delay_seconds=3, temperature=0.1):
    last_error = None
    for attempt in range(1, max_retries + 1):
        _sarvam_pace()
        try:
            response = client.chat.completions(
                model=SARVAM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4000,
                reasoning_effort=None,  # DISABLED -- diagnosed 2026-07-21: no-RAG
                # conditions were failing 95% of the time vs. 42% with RAG present.
                # All failures showed reasoning_content fallback text that never
                # contained a parseable final answer -- reasoning was consuming the
                # token budget before a formatted response was ever produced,
                # worse when there was no retrieved content to anchor to. Disabling
                # trades some "fairness" (vs. Llama3/Hex-1's free-form generation)
                # for actually having usable data -- document this as a methods
                # choice, not silently.
            )
            content = response.choices[0].message.content
            if not content:
                content = getattr(response.choices[0].message, "reasoning_content", None)
            if not content:
                raise ValueError("Sarvam returned empty content and no reasoning_content fallback")
            return content.strip()
        except Exception as e:
            last_error = e
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            wait = delay_seconds * (3 ** (attempt - 1)) if is_rate_limit else delay_seconds
            print(f"    [retry {attempt}/{max_retries}] {type(e).__name__}: {e}"
                  f"{'  (rate limit -- backing off ' + str(wait) + 's)' if is_rate_limit else ''}")
            time.sleep(wait)
    raise last_error


# ══════════════════════════════════════════════════════════════
# CONDITION RUNNER — shared across all three models
# ══════════════════════════════════════════════════════════════

def run_model(model_name, call_fn, queries, kannada_map, tamil_map, malayalam_map,
              glossary, embedded_corpus, corpus_mean, resume_csv, batch_label=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_suffix = f"_{batch_label}" if batch_label else ""
    output_filename = os.path.join(OUTPUT_DIR, f"tulu_mega_{model_name}{batch_suffix}_{timestamp}.csv")

    completed = load_completed(resume_csv)

    output_file = open(output_filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(output_file)
    writer.writerow(["model", "condition", "id", "display_text", "english_gloss",
                      "list_primary", "list_secondary", "pred_top1", "pred_top2",
                      "pred_top3", "top1_confidence", "hits1", "hits3", "partial_credit",
                      "top1_reasoning"])
    output_file.flush()

    all_results = {}

    for condition_id, script, priming_tier, use_rag in CONDITIONS:
        print(f"\n{'='*70}\n{model_name.upper()} — CONDITION {condition_id} "
              f"(script={script}, priming={priming_tier}, RAG={use_rag})\n{'='*70}")

        get_display_text = get_display_text_fn(script, kannada_map, tamil_map, malayalam_map)
        priming_block = None
        if priming_tier:
            script_key = "romanized" if script == "romanized" else script
            priming_block = build_priming_block(glossary, script_key, priming_tier)

        hits1_count = hits3_count = n_resumed = n_fresh = 0

        for query in queries:
            qid_str = str(query["id"])
            prior = completed.get((condition_id, qid_str))

            if prior is not None:
                n_resumed += 1
                cats = [prior["pred_top1"], prior["pred_top2"], prior["pred_top3"]]
                hits1 = prior["hits1"] == "True"
                hits3 = prior["hits3"] == "True"
                partial = prior["partial_credit"] == "True"
                hits1_count += hits1
                hits3_count += hits3
                mark = "✓" if hits1 else ("~" if hits3 else "✗")
                print(f"  [{query['id']:>2}] {cats[0]:<22} {mark}  [resumed]")
                writer.writerow([model_name, condition_id, query["id"], prior["display_text"],
                                  prior["english_gloss"], prior["list_primary"], prior["list_secondary"],
                                  cats[0], cats[1], cats[2], prior["top1_confidence"],
                                  hits1, hits3, partial, prior["top1_reasoning"]])
                output_file.flush()
                continue

            n_fresh += 1
            display_text = get_display_text(query)
            retrieved = retrieve(display_text, embedded_corpus, corpus_mean) if use_rag else []
            prompt = build_unified_prompt(script, display_text, retrieved, priming_block, use_rag)

            top3 = [{"category": "ERROR", "confidence": 0.0, "reasoning": "generation failed"}] * 3
            try:
                raw = call_fn(prompt)
                top3 = validate_categories_safe(extract_top3(raw))
            except Exception as e:
                print(f"  GAVE UP on id {query['id']}: {e}")

            cats = [p["category"] for p in top3]
            hits1, hits3, partial = score_hits(cats, query["list_primary"], query["list_secondary"])
            hits1_count += hits1
            hits3_count += hits3
            mark = "✓" if hits1 else ("~" if hits3 else "✗")
            print(f"  [{query['id']:>2}] {cats[0]:<22} {mark}  expected: {query['list_primary']:<20}")

            writer.writerow([model_name, condition_id, query["id"], display_text,
                              query["english_gloss"], query["list_primary"], query["list_secondary"],
                              cats[0], cats[1], cats[2], f"{top3[0].get('confidence',0):.2f}",
                              hits1, hits3, partial, top3[0].get("reasoning", "")])
            output_file.flush()

        n = len(queries)
        result = {"hits1": hits1_count/n*100, "hits3": hits3_count/n*100}
        all_results[condition_id] = result
        print(f"\n  {condition_id}: Hits@1 {result['hits1']:.1f}%  Hits@3 {result['hits3']:.1f}%  "
              f"({n_resumed} resumed, {n_fresh} fresh)")

    output_file.close()
    print(f"\n{model_name.upper()} complete. Saved to: {output_filename}")
    return all_results, output_filename


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    if not preflight_check():
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_queries, kannada_map, tamil_map, malayalam_map = load_consolidated_queries()
    glossary = load_glossary()

    print("\nLoading corpus and embedding (E5)...")
    passages = load_kannada_corpus(path=KANNADA_CORPUS_PATH)
    embedded_corpus = embed_corpus(passages)
    if USE_MEAN_CENTERING:
        embedded_corpus, corpus_mean = mean_center(embedded_corpus)
    else:
        corpus_mean = np.zeros_like(embedded_corpus[0]["embedding"])
        print("Mean-centering disabled (matches current pipeline config).")

    # Run in 3 batches of 20 -- if a long run gets interrupted, whichever
    # batches already fully completed (across all 3 models) are safe and
    # usable on their own, rather than needing the entire 60-sentence x
    # 3-model run to finish in one sitting.
    all_batch_results = {}
    for batch_num, (lo, hi) in enumerate(BATCHES, start=1):
        batch_queries = [q for q in all_queries if lo <= q["id"] <= hi]
        batch_label = f"batch{batch_num}_ids{lo}-{hi}"

        print(f"\n{'#'*80}\n# BATCH {batch_num} OF {len(BATCHES)} -- ids {lo}-{hi} "
              f"({len(batch_queries)} queries x 19 conditions x 3 models)\n{'#'*80}")

        batch_results = {}

        # ── SARVAM (first -- currently under active debugging, fastest to iterate on) ──
        if SARVAM_API_KEY == "PASTE_YOUR_KEY_HERE":
            print("\nSARVAM_API_KEY not set -- skipping Sarvam entirely.")
        else:
            client = get_sarvam_client()
            sarvam_call_fn = lambda prompt: call_sarvam(client, prompt)
            batch_results["sarvam"], _ = run_model(
                "sarvam", sarvam_call_fn, batch_queries, kannada_map, tamil_map, malayalam_map,
                glossary, embedded_corpus, corpus_mean, RESUME_FROM["sarvam"],
                batch_label=batch_label,
            )

        # ── HEX-1 (second) ──
        batch_results["hex1"], _ = run_model(
            "hex1", call_hex1, batch_queries, kannada_map, tamil_map, malayalam_map,
            glossary, embedded_corpus, corpus_mean, RESUME_FROM["hex1"],
            batch_label=batch_label,
        )

        # ── LLAMA3 (last -- most reliable, no reason to front-load it) ──
        batch_results["llama3"], _ = run_model(
            "llama3", call_llama3, batch_queries, kannada_map, tamil_map, malayalam_map,
            glossary, embedded_corpus, corpus_mean, RESUME_FROM["llama3"],
            batch_label=batch_label,
        )

        all_batch_results[batch_label] = batch_results

        print(f"\n{'='*70}\nBATCH {batch_num} OF {len(BATCHES)} COMPLETE (ids {lo}-{hi}) "
              f"-- results saved, safe even if the run stops here.\n{'='*70}")

    # ── FINAL SUMMARY -- combined across all batches ──
    print(f"\n{'='*80}\nFINAL COMPARISON — ALL CONDITIONS x ALL MODELS x ALL BATCHES\n{'='*80}")
    for batch_label, batch_results in all_batch_results.items():
        print(f"\n--- {batch_label} ---")
        header = f"{'Condition':<12}" + "".join(f"{m:>12}" for m in batch_results)
        print(header)
        for cond_id, _, _, _ in CONDITIONS:
            row = f"{cond_id:<12}"
            for m in batch_results:
                r = batch_results[m].get(cond_id)
                row += f"{r['hits1']:>11.1f}%" if r else f"{'--':>12}"
            print(row)


if __name__ == "__main__":
    main()