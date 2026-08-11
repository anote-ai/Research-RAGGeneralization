"""
TULU LEGAL — CRAG-STYLE RELEVANCE VERIFICATION (STAGE 2), MEGA-PIPELINE PORT
================================================================================
Ports tulu-v5/tulu_crag_stage2.py onto the mega-pipeline engine, same process
used for tulu_crag_stage3.py. Same mitigation idea: before injecting EACH
retrieved passage, a separate, cheap LLM call judges whether it actually
describes the SAME situation as the query (RELEVANT / NOT_RELEVANT). Passages
judged NOT_RELEVANT are discarded; if none of the top-k survive, falls back
to the no-RAG path.

Two conditions, same as the tulu-v5 original -- NEITHER uses any vocabulary
priming for the judge (that's what stage3 adds on top of this):
  - CRAG_judge_kannada:        judge sees the Kannada-script situation text.
                                Realistic deployment condition, no
                                English gloss available.
  - CRAG_judge_english_gloss:  judge sees the English gloss instead.
                                Upper-bound condition -- isolates whether
                                judge quality (vs. the CRAG mechanism
                                itself) is the limiting factor.
Only the JUDGE's input text varies between the two conditions. The classifier
always sees Condition D_full's setup (Kannada script, full Tulu priming, RAG
gated by the judge) regardless of what the judge saw -- same as stage3's port,
and matching the tulu-v5 original's classifier prompt (which also always used
its own priming block regardless of which judge variant was running).

DIFFERENCES FROM THE TULU-V5 ORIGINAL (same rationale as the stage3 port):
  - Runs on mega-pipeline's own data/tulu_legal_20_consolidated.jsonl (the
    20-of-60 sentences whose id + english_gloss exactly match tulu-v5's
    original 20), NOT tulu-v5's own legal-20 files -- the Kannada-script
    wording differs meaningfully between the two projects' files even for
    the "same" sentence.
  - Classifier priming built with tulu_mega_pipeline's own
    build_priming_block/build_unified_prompt (the house style used by every
    other condition in the mega pipeline) instead of the old, now-superseded
    build_static_prefix/build_call_prompt engine path.
  - llama3 only, matching the original single-model CRAG experiment.
  - Output goes to outputs/, not the current working directory.

Requires tulu_rag_engine.py and tulu_mega_pipeline.py in the same folder,
Ollama running with llama3 pulled.
"""

import sys
import os
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tulu_rag_engine import (
    load_kannada_corpus,
    embed_corpus,
    retrieve,
    extract_top3,
    score_hits,
    USE_MEAN_CENTERING,
    mean_center,
)
from tulu_mega_pipeline import (
    DATA_DIR,
    OUTPUT_DIR,
    KANNADA_CORPUS_PATH,
    load_consolidated_queries,
    load_glossary,
    build_priming_block,
    build_unified_prompt,
    validate_categories_safe,
    call_llama3,
    GLOSSARY_PATH,
)

LEGAL_20_PATH = os.path.join(DATA_DIR, "tulu_legal_20_consolidated.jsonl")

JUDGE_MODEL_TEMPERATURE = 0.0  # deterministic -- this is a binary judgment, not creative generation


def build_judge_prompt(situation_text, passage_text, case_name):
    return (
        "You are checking whether a retrieved legal case describes the SAME "
        "situation as a person's legal complaint.\n\n"
        f"PERSON'S SITUATION: \"{situation_text}\"\n\n"
        f"RETRIEVED CASE ({case_name}):\n{passage_text[:1500]}\n\n"
        "Does this retrieved case describe the SAME kind of situation as the "
        "person's complaint above -- the same general legal problem, even if "
        "specific facts differ? Answer with EXACTLY one word: "
        "RELEVANT or NOT_RELEVANT. No explanation, no punctuation, just the word."
    )


def judge_relevance(situation_text, passage):
    prompt = build_judge_prompt(situation_text, passage["text"], passage.get("case_name", "?"))
    try:
        raw = call_llama3(prompt, temperature=JUDGE_MODEL_TEMPERATURE)
    except Exception as e:
        print(f"    Judge call failed: {e} -- defaulting to NOT_RELEVANT (safe default)")
        return False
    verdict = raw.strip().upper()
    return "RELEVANT" in verdict and "NOT_RELEVANT" not in verdict


def run_crag_condition(condition_name, queries, kannada_map, embedded_corpus, corpus_mean,
                        judge_situation_fn, classifier_priming_block, writer, output_file):
    """
    judge_situation_fn(query) -> text to show the JUDGE (not the classifier).
    The classifier always sees Condition D_full's prompt (Kannada script,
    full Tulu priming) built via build_unified_prompt, gated by whether the
    judge let any passages survive -- only the judge's input text varies
    between the two conditions run in main().
    """
    print(f"\n{'━'*65}\nRUNNING: {condition_name}\n{'━'*65}")

    hits1_count = hits3_count = n_fell_back = 0

    for query in queries:
        display_text = kannada_map[query["id"]]
        retrieved = retrieve(display_text, embedded_corpus, corpus_mean)

        judge_text = judge_situation_fn(query)
        surviving = []
        verdicts = []
        for p in retrieved:
            is_relevant = judge_relevance(judge_text, p)
            verdicts.append((p.get("case_name", "?"), is_relevant))
            if is_relevant:
                surviving.append(p)

        used_rag = len(surviving) > 0
        if not used_rag:
            n_fell_back += 1

        prompt = build_unified_prompt("kannada", display_text, surviving,
                                       classifier_priming_block, used_rag)

        top3 = [{"category": "ERROR", "confidence": 0.0, "reasoning": "generation failed"}] * 3
        try:
            raw = call_llama3(prompt)
            top3 = validate_categories_safe(extract_top3(raw))
        except Exception as e:
            print(f"  GAVE UP on id {query['id']}: {e}")

        cats = [p["category"] for p in top3]
        hits1, hits3, partial = score_hits(cats, query["list_primary"], query["list_secondary"])
        hits1_count += hits1
        hits3_count += hits3

        mark = "✓" if hits1 else ("~" if hits3 else "✗")
        verdict_str = ", ".join(f"{name[:20]}:{'REL' if v else 'NOT'}" for name, v in verdicts)
        fallback_note = " [FELL BACK TO NO-RAG]" if not used_rag else f" [{len(surviving)} passage(s) survived]"
        print(f"  [{query['id']:>2}] {cats[0]:<22} {mark}  expected: {query['list_primary']:<20}"
              f"{fallback_note}")
        print(f"       judge verdicts: {verdict_str}")

        writer.writerow([condition_name, query["id"], display_text, query["english_gloss"],
                          query["list_primary"], query["list_secondary"],
                          cats[0], cats[1], cats[2], f"{top3[0].get('confidence',0):.2f}",
                          hits1, hits3, partial, used_rag, len(surviving), top3[0].get("reasoning", "")])
        output_file.flush()

    n = len(queries)
    print(f"\n  {condition_name}: Hits@1 {hits1_count/n*100:.1f}% ({hits1_count}/{n})  "
          f"Hits@3 {hits3_count/n*100:.1f}% ({hits3_count}/{n})  "
          f"Fell back to no-RAG on {n_fell_back}/{n} queries")
    return {"hits1": hits1_count/n*100, "hits3": hits3_count/n*100, "n_fell_back": n_fell_back}


def preflight_check():
    print("=" * 70)
    print("PRE-FLIGHT CHECK")
    print("=" * 70)
    ok = True
    for path in [LEGAL_20_PATH, KANNADA_CORPUS_PATH, GLOSSARY_PATH]:
        exists = os.path.exists(path)
        print(f"  [{'OK' if exists else 'MISSING'}] {path}")
        ok = ok and exists
    if not ok:
        print("\nABORTING: one or more required files are missing.")
    return ok


def main():
    if not preflight_check():
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(OUTPUT_DIR, f"tulu_crag_stage2_{timestamp}.csv")

    print("\nLoading corpus and embedding (E5)...")
    passages = load_kannada_corpus(path=KANNADA_CORPUS_PATH)
    embedded_corpus = embed_corpus(passages)
    if USE_MEAN_CENTERING:
        embedded_corpus, corpus_mean = mean_center(embedded_corpus)
    else:
        corpus_mean = np.zeros_like(embedded_corpus[0]["embedding"])
        print("Mean-centering disabled (matches current pipeline config).")

    queries, kannada_map, _tamil_map, _malayalam_map = load_consolidated_queries(path=LEGAL_20_PATH)
    glossary = load_glossary()
    print(f"Testing {len(queries)} sentences (the subset of the 60 whose id + "
          f"english_gloss match tulu-v5's original 20).")

    # Classifier always gets full Tulu priming -- matches Condition D_full,
    # and mirrors the tulu-v5 original always giving the classifier its own
    # priming block regardless of which judge condition was running.
    classifier_priming_block = build_priming_block(glossary, "kannada", "full")

    output_file = open(output_filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(output_file)
    writer.writerow(["condition", "id", "display_text", "english_gloss",
                      "list_primary", "list_secondary", "pred_top1", "pred_top2",
                      "pred_top3", "top1_confidence", "hits1", "hits3", "partial_credit",
                      "used_rag", "n_passages_survived", "top1_reasoning"])
    output_file.flush()

    results = {}

    # Variant 1: judge sees Kannada-script text only -- realistic deployment condition.
    results["CRAG_judge_kannada"] = run_crag_condition(
        "CRAG_judge_kannada", queries, kannada_map, embedded_corpus, corpus_mean,
        judge_situation_fn=lambda q: kannada_map[q["id"]],
        classifier_priming_block=classifier_priming_block,
        writer=writer, output_file=output_file,
    )

    # Variant 2: judge sees English gloss -- upper-bound / judge-quality-isolated condition.
    results["CRAG_judge_english_gloss"] = run_crag_condition(
        "CRAG_judge_english_gloss", queries, kannada_map, embedded_corpus, corpus_mean,
        judge_situation_fn=lambda q: q["english_gloss"],
        classifier_priming_block=classifier_priming_block,
        writer=writer, output_file=output_file,
    )

    output_file.close()

    print(f"\n{'='*65}\nFINAL COMPARISON\n{'='*65}")
    print(f"{'Condition':<28} {'Hits@1':>8} {'Hits@3':>8} {'Fell back':>10}")
    print("(tulu-v5 original-experiment reference numbers below used different "
          "Kannada-script wording for the same sentences -- not directly comparable, "
          "context only)")
    print(f"{'D (pure RAG, tulu-v5 ref)':<28} {'25.0%':>8} {'65.0%':>8} {'0/20':>10}")
    print(f"{'E (pure no-RAG, tulu-v5 ref)':<28} {'80.0%':>8} {'95.0%':>8} {'20/20':>10}")
    for name, r in results.items():
        n = len(queries)
        print(f"{name:<28} {r['hits1']:>7.1f}% {r['hits3']:>7.1f}% {r['n_fell_back']:>7}/{n}")
    print(f"\nResults saved to: {output_filename}")
    print("\nHOW TO READ THIS:")
    print("  - If CRAG_judge_kannada beats the pure-RAG reference substantially,")
    print("    relevance verification helps even with a same-resource-quality")
    print("    judge -- a real, deployable win.")
    print("  - If CRAG_judge_kannada stays low but CRAG_judge_english_gloss")
    print("    recovers close to the no-RAG ceiling, the CRAG mechanism works")
    print("    but the judge itself needs a translation step to be reliable on")
    print("    Tulu -- an important, honest limitation to report.")
    print("  - If BOTH stay low, the judge isn't the bottleneck by itself --")
    print("    see tulu_crag_stage3.py for whether targeted vocabulary priming")
    print("    fixes the Kannada-only judge instead.")


if __name__ == "__main__":
    main()
