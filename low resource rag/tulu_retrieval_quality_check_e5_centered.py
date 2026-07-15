"""
TULU LEGAL — RETRIEVAL QUALITY CHECK, E5 WITH MEAN-CENTERING
=================================================================
Direct companion to tulu_retrieval_quality_check.py (raw E5, no centering).
Same corpus, same 20 queries, same category-hit scoring -- but this time
mean-centering IS applied, using the pipeline's own mean_center() function.

Answers: does centering help E5 (cleans up the mild Amith Ramachandra
over-representation from the raw run), hurt it (same pathology we saw
with nomic), or do nothing?

Requires the UPDATED tulu_legal_rag_v7.py (the one with E5 support +
USE_MEAN_CENTERING toggle) in the same folder. Uses its own mean_center()
and get_embedding() directly so this stays consistent with the real
pipeline rather than reimplementing centering separately.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tulu_legal_rag_v8 import (
    load_kannada_corpus,
    load_tulu_queries,
    load_kannada_script_tulu,
    embed_corpus,
    mean_center,
    get_embedding,
    cosine_similarity,
    _init_embed_model,
    _embed_model_name,
)


def main():
    print("=" * 70)
    print("RETRIEVAL QUALITY CHECK — E5, WITH mean-centering")
    print("=" * 70)

    passages = load_kannada_corpus()
    print(f"\nEmbedding {len(passages)} corpus chunks...")
    embedded = embed_corpus(passages)  # is_query=False by default -- correct for passages
    print(f"Embedding backend in use: {_embed_model_name}")

    print("\nApplying mean-centering...")
    centered, corpus_mean = mean_center(embedded)

    tulu_queries = load_tulu_queries()
    kannada_script = load_kannada_script_tulu()
    if kannada_script is None:
        print("Kannada-script file not found -- cannot run this test.")
        return

    gloss_by_id = {q["id"]: q for q in tulu_queries}

    print("\n" + "-" * 70)
    print("PER-QUERY: does the top-1 / top-3 retrieved chunk match the true category?")
    print("(random chance across 8 LIST categories ~= 12.5%)")
    print("-" * 70)

    hits1, hits3 = 0, 0
    top1_doc_counts = {}
    for qid, text in kannada_script.items():
        q = gloss_by_id[qid]
        primary = q["list_primary"]
        secondary = q.get("list_secondary", "")
        true_cats = {c for c in [primary, secondary] if c}

        query_vec = get_embedding(text, is_query=True) - corpus_mean
        scored = sorted(
            centered,
            key=lambda p: cosine_similarity(query_vec, p["embedding"]),
            reverse=True,
        )
        top3 = scored[:3]
        top1_cats = set(top3[0]["list_categories"])
        top3_cats = set().union(*(set(p["list_categories"]) for p in top3))

        hit1 = bool(true_cats & top1_cats)
        hit3 = bool(true_cats & top3_cats)
        hits1 += hit1
        hits3 += hit3

        top1_doc = top3[0]["case_name"]
        top1_doc_counts[top1_doc] = top1_doc_counts.get(top1_doc, 0) + 1

        top1_score = cosine_similarity(query_vec, top3[0]["embedding"])
        print(f"  Q{qid:>2} [{primary:<20}] top1={top1_doc[:30]:<30} "
              f"score={top1_score:.4f} cats={list(top1_cats)[:2]} "
              f"{'HIT' if hit1 else 'miss'}")

    print("\n" + "-" * 70)
    print(f"CENTERED retrieval accuracy (E5 + mean-centering):")
    print(f"  Category Hits@1: {hits1}/20 ({100*hits1/20:.1f}%)")
    print(f"  Category Hits@3: {hits3}/20 ({100*hits3/20:.1f}%)")
    print(f"  Random baseline (8 categories): ~12.5%")
    print(f"\n  Top-1 document diversity: {len(top1_doc_counts)} distinct docs won top-1")
    for doc, count in sorted(top1_doc_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:2}/20  {doc}")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("COMPARE TO RAW E5 (tulu_retrieval_quality_check.py, no centering):")
    print("  Raw E5:      Hits@1 40.0%  Hits@3 75.0%  (10 distinct docs won top-1)")
    print(f"  Centered E5: Hits@1 {100*hits1/20:.1f}%  Hits@3 {100*hits3/20:.1f}%  "
          f"({len(top1_doc_counts)} distinct docs won top-1)")
    print("")
    print("  - If centered Hits@1/@3 are HIGHER and/or doc diversity increased")
    print("    -> centering genuinely helps E5. Set USE_MEAN_CENTERING = True.")
    print("  - If centered numbers are the SAME or WORSE, and/or diversity")
    print("    dropped (fewer distinct docs, or one doc starting to dominate)")
    print("    -> centering isn't helping E5, possibly repeating the nomic")
    print("    pathology in a milder form. Keep USE_MEAN_CENTERING = False.")
    print("=" * 70)


if __name__ == "__main__":
    main()
