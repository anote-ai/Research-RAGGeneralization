"""
TULU LEGAL — STAGE 2 MITIGATION: CRAG-STYLE RELEVANCE VERIFICATION
=======================================================================
Stage 1 (confidence-gated retrieval, pure similarity threshold) was tested
via offline sweep on existing data and found NOT to recover the collapse --
accuracy climbed monotonically only as RAG was excluded almost entirely,
never beating the 80% no-RAG ceiling at any threshold. That result matched
the report's own diagnostic: "if it stays low even when only high-similarity
passages pass, the problem is genuine distraction... proceed to Stage 2."

This script implements Stage 2: before injecting EACH retrieved passage,
a separate, cheap LLM call judges whether it actually describes the SAME
situation as the query (RELEVANT / NOT_RELEVANT). Passages judged
NOT_RELEVANT are discarded. If none of the top-k survive, falls back to
the no-RAG path (build_call_prompt(no_rag=True)) -- the same path that
scores 80% on its own.

Runs TWO judge variants so we can see whether judge reliability itself is
a bottleneck (per the mitigation report's caveat that the judge is itself
an LLM call on a low-resource language and may inherit the same weakness):
  - JUDGE ON KANNADA-SCRIPT TEXT ONLY: realistic deployment condition, no
    English gloss available (matches Condition D's actual test setup).
  - JUDGE ON ENGLISH GLOSS: upper-bound condition -- what CRAG could
    achieve with a maximally reliable relevance judge, IF a gloss/
    translation step existed. Not representative of a real zero-resource
    deployment on its own, but isolates whether judge quality vs. the
    CRAG mechanism itself is the limiting factor.

Requires tulu_legal_rag_v7.py (the E5 version, USE_MEAN_CENTERING=False)
in the same folder, Ollama running.
"""

import sys
import os
import csv
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama
import numpy as np
from tulu_legal_rag_v8 import (
    load_kannada_corpus,
    load_tulu_queries,
    load_kannada_script_tulu,
    embed_corpus,
    retrieve,
    build_static_prefix,
    build_call_prompt,
    extract_top3,
    validate_categories,
    score_hits,
    USE_MEAN_CENTERING,
    mean_center,
)

JUDGE_MODEL_TEMPERATURE = 0.0  # deterministic -- this is a binary judgment, not creative generation


def call_model_with_retry(prompt, max_retries=3, delay_seconds=2, temperature=0.1):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.generate(
                model="llama3", prompt=prompt,
                options={"temperature": temperature, "repeat_penalty": 1.1},
            )
            return response["response"].strip()
        except Exception as e:
            last_error = e
            print(f"    [retry {attempt}/{max_retries}] {type(e).__name__}: {e}")
            time.sleep(delay_seconds)
    raise last_error


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


def judge_relevance(situation_text, passage, use_retry=True):
    prompt = build_judge_prompt(situation_text, passage["text"], passage.get("case_name", "?"))
    try:
        raw = call_model_with_retry(prompt, temperature=JUDGE_MODEL_TEMPERATURE) if use_retry \
            else ollama.generate(model="llama3", prompt=prompt,
                                  options={"temperature": JUDGE_MODEL_TEMPERATURE})["response"].strip()
    except Exception as e:
        print(f"    Judge call failed: {e} -- defaulting to NOT_RELEVANT (safe default)")
        return False
    verdict = raw.strip().upper()
    return "RELEVANT" in verdict and "NOT_RELEVANT" not in verdict


def run_crag_condition(condition_name, queries, kannada_map, embedded_corpus, corpus_mean,
                        judge_situation_fn, writer, output_file):
    """
    judge_situation_fn(query) -> text to show the JUDGE (not the classifier)
    The classifier always sees the real Condition-D-style prompt (Kannada
    script, no gloss) regardless of what the judge sees -- only the judge's
    input text varies between the two variants we run.
    """
    print(f"\n{'━'*65}\nRUNNING: {condition_name}\n{'━'*65}")
    static_prefix = build_static_prefix(include_priming=True, kannada_script_mode=True)

    hits1_count = hits3_count = 0
    n_fell_back = 0

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

        prompt = build_call_prompt(
            static_prefix, display_text, english_gloss=None,
            retrieved=surviving, script_label="Kannada script",
            kannada_script_mode=True, no_rag=not used_rag,
        )

        top3 = [{"category": "ERROR", "confidence": 0.0, "reasoning": "generation failed"}] * 3
        try:
            raw = call_model_with_retry(prompt)
            top3 = validate_categories(extract_top3(raw))
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


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"tulu_crag_stage2_{timestamp}.csv"

    print("Loading corpus and embedding (E5)...")
    passages = load_kannada_corpus()
    embedded_corpus = embed_corpus(passages)
    if USE_MEAN_CENTERING:
        embedded_corpus, corpus_mean = mean_center(embedded_corpus)
    else:
        corpus_mean = np.zeros_like(embedded_corpus[0]["embedding"])
        print("Mean-centering disabled (matches current pipeline config).")

    queries = load_tulu_queries()
    kannada_map = load_kannada_script_tulu()

    output_file = open(output_filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(output_file)
    writer.writerow(["condition", "id", "display_text", "english_gloss",
                      "list_primary", "list_secondary", "pred_top1", "pred_top2",
                      "pred_top3", "top1_confidence", "hits1", "hits3", "partial_credit",
                      "used_rag", "n_passages_survived", "top1_reasoning"])
    output_file.flush()

    results = {}

    # Variant 1: judge sees Kannada-script text only -- realistic deployment condition
    results["CRAG_judge_kannada"] = run_crag_condition(
        "CRAG_judge_kannada", queries, kannada_map, embedded_corpus, corpus_mean,
        judge_situation_fn=lambda q: kannada_map[q["id"]],
        writer=writer, output_file=output_file,
    )

    # Variant 2: judge sees English gloss -- upper-bound / judge-quality-isolated condition
    results["CRAG_judge_english_gloss"] = run_crag_condition(
        "CRAG_judge_english_gloss", queries, kannada_map, embedded_corpus, corpus_mean,
        judge_situation_fn=lambda q: q["english_gloss"],
        writer=writer, output_file=output_file,
    )

    output_file.close()

    print(f"\n{'='*65}\nFINAL COMPARISON\n{'='*65}")
    print(f"{'Condition':<28} {'Hits@1':>8} {'Hits@3':>8} {'Fell back':>10}")
    print(f"{'D (pure RAG, reference)':<28} {'25.0%':>8} {'65.0%':>8} {'0/20':>10}")
    print(f"{'E (pure no-RAG, reference)':<28} {'80.0%':>8} {'95.0%':>8} {'20/20':>10}")
    for name, r in results.items():
        print(f"{name:<28} {r['hits1']:>7.1f}% {r['hits3']:>7.1f}% {r['n_fell_back']:>7}/20")
    print(f"\nResults saved to: {output_filename}")
    print("\nHOW TO READ THIS:")
    print("  - If CRAG_judge_kannada beats 25% substantially, relevance")
    print("    verification helps even with a same-resource-quality judge --")
    print("    a real, deployable win.")
    print("  - If CRAG_judge_kannada stays low but CRAG_judge_english_gloss")
    print("    recovers close to 80%, the CRAG mechanism works but the judge")
    print("    itself needs a translation step to be reliable on Tulu --")
    print("    an important, honest limitation to report.")
    print("  - If BOTH stay low, the judge isn't the bottleneck and Stage 3")
    print("    (disagreement guard) or a different approach is needed.")


if __name__ == "__main__":
    main()
