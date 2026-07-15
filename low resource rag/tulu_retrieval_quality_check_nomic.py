"""
TULU LEGAL — RAW RETRIEVAL QUALITY CHECK (nomic-embed-text)
================================================================
Direct companion to tulu_retrieval_quality_check.py (the E5 version).
Same methodology, same corpus, same 20 queries, same scoring -- but using
nomic-embed-text via Ollama instead of E5, and with NO mean-centering, NO
quota, NO MMR (the actual v7 pipeline's raw scores would normally go
through all three -- this strips them out so it's a fair apples-to-apples
comparison against the E5 raw numbers).

This answers the open question: was nomic's raw ranking ever actually
broken, or did the mean-centering/quota/MMR machinery break something
that raw nomic would have gotten right?

Run in the same folder as tulu_legal_rag_v7.py, with Ollama running and
nomic-embed-text pulled.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tulu_legal_rag_v7 import (
    load_kannada_corpus,
    load_tulu_queries,
    load_kannada_script_tulu,
    get_embedding,       # nomic via Ollama, exactly as the real pipeline calls it
    cosine_similarity,
)


def main():
    print("=" * 70)
    print("RAW RETRIEVAL QUALITY CHECK — nomic-embed-text, no centering/quota/MMR")
    print("=" * 70)

    passages = load_kannada_corpus()
    print(f"\nEmbedding {len(passages)} corpus chunks with nomic-embed-text...")
    for i, p in enumerate(passages):
        p["embedding"] = get_embedding(p["text"])
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(passages)}")

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
    for qid, text in kannada_script.items():
        q = gloss_by_id[qid]
        primary = q["list_primary"]
        secondary = q.get("list_secondary", "")
        true_cats = {c for c in [primary, secondary] if c}

        query_vec = get_embedding(text)
        scored = sorted(
            passages,
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

        top1_score = cosine_similarity(query_vec, top3[0]["embedding"])
        print(f"  Q{qid:>2} [{primary:<20}] top1={top3[0]['case_name'][:30]:<30} "
              f"score={top1_score:.4f} cats={list(top1_cats)[:2]} "
              f"{'HIT' if hit1 else 'miss'}")

    print("\n" + "-" * 70)
    print(f"RAW retrieval accuracy (nomic, no pipeline fixes applied):")
    print(f"  Category Hits@1: {hits1}/20 ({100*hits1/20:.1f}%)")
    print(f"  Category Hits@3: {hits3}/20 ({100*hits3/20:.1f}%)")
    print(f"  Random baseline (8 categories): ~12.5%")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("COMPARE THIS OUTPUT TO tulu_retrieval_quality_check.py (E5):")
    print("  E5 raw:    Hits@1 40.0%  Hits@3 75.0%  (10 distinct docs won top-1)")
    print("  nomic raw: Hits@1 ???    Hits@3 ???    (check top1 doc diversity above)")
    print("")
    print("  - If nomic raw is ALSO well above 12.5% and ALSO diverse across")
    print("    documents -> the embedding model was never the problem. The")
    print("    mean-centering/quota/MMR machinery broke something that raw")
    print("    nomic already had right. Fix: revisit or drop those fixes,")
    print("    keep nomic.")
    print("  - If nomic raw is near-random OR one document (e.g. Neelamegham)")
    print("    still dominates top-1 even without centering -> the embedding")
    print("    model genuinely matters, and E5 is a real improvement worth")
    print("    carrying into the full pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
