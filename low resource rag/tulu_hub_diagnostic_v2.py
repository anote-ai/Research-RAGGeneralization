"""
TULU LEGAL — HUB DIAGNOSTIC, ROUND 2
======================================
Round 1 ruled out the "Neelamegham's vector collapses to near-zero" theory
(its norm was unremarkable, mid-pack). This round tests the refined
hypothesis instead: are KANNADA-SCRIPT queries too weakly differentiated
from each other, such that whichever document happens to sit in their
general direction wins almost every query -- regardless of that document's
own vector being anything special?

This time we test NEELAMEGHAM SPECIFICALLY (not an auto-picked doc) using
KANNADA-SCRIPT queries specifically (not romanized) -- matching exactly
what Condition D actually does.

Run in the same folder as tulu_legal_rag_v7.py, with your embedding
backend running.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tulu_legal_rag_v7 import (
    load_kannada_corpus,
    load_tulu_queries,
    load_kannada_script_tulu,
    embed_corpus,
    mean_center,
    get_embedding,
    cosine_similarity,
)


def l2_norm(v):
    return float(np.linalg.norm(v))


def main():
    print("=" * 70)
    print("HUB DIAGNOSTIC ROUND 2 — Neelamegham + Kannada-script queries")
    print("=" * 70)

    passages = load_kannada_corpus()
    embedded = embed_corpus(passages)
    centered, corpus_mean = mean_center(embedded)

    tulu_queries = load_tulu_queries()
    kannada_script = load_kannada_script_tulu()
    if kannada_script is None:
        print("Kannada-script file not found -- cannot run this test.")
        return

    gloss_by_id = {q["id"]: q["english_gloss"] for q in tulu_queries}

    # ── Pairwise query-to-query similarity: how tightly do Kannada-script
    #    queries actually cluster together, compared to romanized ones? ──
    def centered_vecs(text_by_id, raw_text_lookup):
        vecs = {}
        for qid in text_by_id:
            text = raw_text_lookup(qid)
            vecs[qid] = get_embedding(text) - corpus_mean
        return vecs

    kannada_vecs = {qid: get_embedding(text) - corpus_mean for qid, text in kannada_script.items()}
    romanized_vecs = {q["id"]: get_embedding(q["tulu"]) - corpus_mean for q in tulu_queries}

    def avg_pairwise_sim(vecs):
        ids = list(vecs.keys())
        sims = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sims.append(cosine_similarity(vecs[ids[i]], vecs[ids[j]]))
        return float(np.mean(sims)), float(np.std(sims))

    kan_mean, kan_std = avg_pairwise_sim(kannada_vecs)
    rom_mean, rom_std = avg_pairwise_sim(romanized_vecs)

    print("\n" + "-" * 70)
    print("QUERY-TO-QUERY SIMILARITY (how much do queries resemble EACH OTHER)")
    print("High mean = queries aren't well differentiated from one another")
    print("-" * 70)
    print(f"  Kannada-script queries:  mean pairwise sim = {kan_mean:.4f}  (std {kan_std:.4f})")
    print(f"  Romanized queries:       mean pairwise sim = {rom_mean:.4f}  (std {rom_std:.4f})")

    # ── Now: Neelamegham specifically, raw vs centered, Kannada-script queries ──
    neel_raw = [p for p in embedded if p["case_name"].startswith("Neelamegham")]
    neel_centered = [p for p in centered if p["case_name"].startswith("Neelamegham")]

    if not neel_raw:
        print("\nCouldn't find Neelamegham chunks -- check case_name string match.")
        return

    print("\n" + "-" * 70)
    print(f"NEELAMEGHAM: raw vs centered similarity, KANNADA-SCRIPT queries")
    print(f"({len(neel_raw)} chunks)")
    print("-" * 70)
    for qid, text in kannada_script.items():
        raw_q = get_embedding(text)
        centered_q = raw_q - corpus_mean
        raw_sim = max(cosine_similarity(raw_q, c["embedding"]) for c in neel_raw)
        centered_sim = max(cosine_similarity(centered_q, c["embedding"]) for c in neel_centered)
        flag = "<-- INFLATED" if centered_sim - raw_sim > 0.2 else ""
        gloss = gloss_by_id.get(qid, "?")[:40]
        print(f"  Query {qid:>2} ({gloss:40}) raw={raw_sim:.4f}  centered={centered_sim:.4f}  {flag}")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("  - If Kannada-script pairwise similarity is much higher than")
    print("    romanized -> queries genuinely aren't differentiating from")
    print("    each other for this script/language, independent of any one")
    print("    document. That's the real bottleneck, and it means better")
    print("    hub-suppression won't fix Condition D -- a better embedding")
    print("    model for Kannada-script Tulu might.")
    print("  - If Neelamegham's centered similarity is inflated (much higher")
    print("    than raw) specifically for Kannada-script queries -> the")
    print("    original centering hypothesis holds, just specific to this")
    print("    doc + this query type combination, not documents generally.")
    print("=" * 70)


if __name__ == "__main__":
    main()
