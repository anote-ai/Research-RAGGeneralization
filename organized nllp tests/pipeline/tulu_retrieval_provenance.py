"""
TULU LEGAL — RETRIEVAL PROVENANCE LOGGER
================================================================================
Logs exactly which passages RAG retrieves for every (condition, query) pair,
WITHOUT calling any LLM (no Llama3, no Hex-1, no Sarvam).

WHY THIS EXISTS
----------------
The generation pipeline (tulu_mega_pipeline.py) only saves the model's final
answer + reasoning trace, not the retrieved passages that were shown to the
model to produce it. That gap made it impossible to say, case-by-case, whether
a weird reasoning trace was fact substitution (model fixated on something
really in a retrieved passage) or confabulation (model invented content with
no basis at all) -- exactly the ambiguity flagged in the paper appendix's
methodological note ("the retrieved passages themselves were not preserved in
the experiment logs for this batch").

This script closes that gap on its own, decoupled from any model run: it
embeds the corpus + every query script variant, retrieves top-3 passages for
every condition that uses RAG, and logs the full passage text + score +
source doc + LIST categories per row. Cross-reference this CSV's
(condition, id) pair against a tulu_mega_*.csv row to see precisely what the
model had in front of it when it produced that reasoning trace.

COST
----
Embedding the corpus + all query variants is the only slow part (same cost as
the embedding step inside tulu_mega_pipeline.py). Retrieval itself is just
cosine similarity over already-computed vectors -- no network/API calls, no
Ollama, no LLM inference -- so once embedding finishes this completes in
seconds, not the hours a full 3-model x 19-condition run takes.

REUSES THE REAL PIPELINE'S CODE, NOT A REIMPLEMENTATION
----------------------------------------------------------
- Embedding + retrieve() imported directly from tulu_rag_engine.py
- CONDITIONS table + query/script loading imported directly from
  tulu_mega_pipeline.py
This guarantees the passages logged here are IDENTICAL to what a real
generation run would retrieve for the same (condition, query) -- there is no
second, drifting copy of the retrieval logic to keep in sync.

SETUP: same data/ folder as tulu_mega_pipeline.py, run from the same
pipeline/ folder as tulu_mega_pipeline.py + tulu_rag_engine.py.
"""

import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tulu_rag_engine import (
    load_kannada_corpus,
    embed_corpus,
    retrieve,
    USE_MEAN_CENTERING,
    mean_center,
)
from tulu_mega_pipeline import (
    CONDITIONS,
    KANNADA_CORPUS_PATH,
    OUTPUT_DIR,
    preflight_check,
    load_consolidated_queries,
    get_display_text_fn,
)

TOP_N_TO_LOG = 3


def main():
    if not preflight_check():
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_queries, kannada_map, tamil_map, malayalam_map = load_consolidated_queries()

    print("\nLoading corpus and embedding (E5)...")
    passages = load_kannada_corpus(path=KANNADA_CORPUS_PATH)
    embedded_corpus = embed_corpus(passages)
    if USE_MEAN_CENTERING:
        embedded_corpus, corpus_mean = mean_center(embedded_corpus)
    else:
        corpus_mean = np.zeros_like(embedded_corpus[0]["embedding"])
        print("Mean-centering disabled (matches current pipeline config).")

    rag_conditions = [c for c in CONDITIONS if c[3]]           # use_rag == True
    skipped_conditions = [c[0] for c in CONDITIONS if not c[3]]
    print(f"\n{len(rag_conditions)} RAG conditions to log: "
          f"{', '.join(c[0] for c in rag_conditions)}")
    print(f"Skipping {len(skipped_conditions)} non-RAG conditions (nothing "
          f"retrieved for these): {', '.join(skipped_conditions)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(OUTPUT_DIR, f"tulu_retrieval_provenance_{timestamp}.csv")

    fieldnames = ["condition", "script", "id", "display_text",
                  "list_primary", "list_secondary"]
    for i in range(1, TOP_N_TO_LOG + 1):
        fieldnames += [f"p{i}_score", f"p{i}_case_name", f"p{i}_doc_id",
                        f"p{i}_chunk_index", f"p{i}_list_categories", f"p{i}_text"]

    # Retrieval depends ONLY on the query's display text, i.e. its script --
    # priming tier never touches retrieval. So B / C_sparse / C_full (all
    # "romanized") retrieve identically, as do D_sparse / D_full / F (all
    # "kannada"). Cache per (script, query id) so each distinct retrieval is
    # only computed once and reused across the conditions that share it,
    # rather than repeating the same cosine-similarity search redundantly.
    retrieval_cache = {}  # (script, query_id) -> list of retrieved passage dicts

    total_rows = 0
    total_retrievals = 0

    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for condition_id, script, priming_tier, use_rag in rag_conditions:
            get_display_text = get_display_text_fn(script, kannada_map, tamil_map, malayalam_map)
            print(f"\n{condition_id} (script={script})...")

            for query in all_queries:
                display_text = get_display_text(query)
                cache_key = (script, query["id"])

                if cache_key in retrieval_cache:
                    retrieved = retrieval_cache[cache_key]
                else:
                    retrieved = retrieve(display_text, embedded_corpus, corpus_mean)
                    retrieval_cache[cache_key] = retrieved
                    total_retrievals += 1

                row = {
                    "condition": condition_id,
                    "script": script,
                    "id": query["id"],
                    "display_text": display_text,
                    "list_primary": query["list_primary"],
                    "list_secondary": query["list_secondary"],
                }
                for i in range(1, TOP_N_TO_LOG + 1):
                    if i <= len(retrieved):
                        p = retrieved[i - 1]
                        row[f"p{i}_score"] = f"{p['score']:.4f}"
                        row[f"p{i}_case_name"] = p.get("case_name", "")
                        row[f"p{i}_doc_id"] = p.get("doc_id", "")
                        row[f"p{i}_chunk_index"] = p.get("chunk_index", "")
                        row[f"p{i}_list_categories"] = " | ".join(p.get("list_categories") or [])
                        row[f"p{i}_text"] = p["text"]
                    else:
                        # Shouldn't happen given TOP_K=3 in tulu_rag_engine.py,
                        # but pad rather than crash if the corpus is ever
                        # smaller than top_k.
                        row[f"p{i}_score"] = ""
                        row[f"p{i}_case_name"] = ""
                        row[f"p{i}_doc_id"] = ""
                        row[f"p{i}_chunk_index"] = ""
                        row[f"p{i}_list_categories"] = ""
                        row[f"p{i}_text"] = ""

                writer.writerow(row)
                total_rows += 1

            f.flush()
            print(f"  {condition_id}: {len(all_queries)} rows logged.")

    print(f"\n{'='*70}")
    print(f"DONE. {total_rows} rows written across {len(rag_conditions)} RAG "
          f"conditions x {len(all_queries)} queries.")
    print(f"{total_retrievals} distinct retrievals actually computed "
          f"({total_rows - total_retrievals} reused from cache -- conditions "
          f"sharing a script retrieve identically).")
    print(f"Saved to: {output_filename}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
