"""
TULU LEGAL — HUB COLLAPSE DIAGNOSTIC
======================================
Tests the hypothesis: after mean-centering, does Neelamegham (and the other
"hub" documents) end up with an abnormally small L2 norm compared to the rest
of the corpus? A small norm means the vector's direction is noise-dominated,
which would explain why it shows suspiciously high, suspiciously uniform
cosine similarity to almost every query regardless of topic.

Reuses load_kannada_corpus / embed_corpus / mean_center / get_embedding /
cosine_similarity directly from tulu_legal_rag_v7.py so this stays fully
consistent with how the actual pipeline embeds and centers things.

Run this in the same environment/folder as tulu_legal_rag_v7.py, with your
Ollama (or MuRIL) embedding backend running locally.
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
    _init_embed_model,
    _embed_model_name,
)


def l2_norm(v):
    return float(np.linalg.norm(v))


def main():
    print("=" * 70)
    print("HUB COLLAPSE DIAGNOSTIC")
    print("=" * 70)

    # ── Step 1: load + embed + center the corpus, exactly as v7 does ──
    passages = load_kannada_corpus()
    embedded = embed_corpus(passages)
    print(f"\nEmbedding backend in use: {_embed_model_name}")

    centered, corpus_mean = mean_center(embedded)

    # ── Step 2: L2 norm of every chunk's CENTERED vector, grouped by doc ──
    per_doc_norms = {}
    for p in centered:
        doc = p["case_name"]
        per_doc_norms.setdefault(doc, []).append(l2_norm(p["embedding"]))

    print("\n" + "-" * 70)
    print("MEAN CENTERED-VECTOR NORM PER DOCUMENT (sorted smallest -> largest)")
    print("Smallest norm = closest to corpus centroid = most 'hub-like'")
    print("-" * 70)
    doc_avg = {doc: np.mean(norms) for doc, norms in per_doc_norms.items()}
    for doc, avg_norm in sorted(doc_avg.items(), key=lambda x: x[1]):
        n_chunks = len(per_doc_norms[doc])
        print(f"  {avg_norm:8.4f}  ({n_chunks:3} chunks)  {doc}")

    # For reference: what's a "normal" norm before centering?
    raw_norms = [l2_norm(p["embedding"]) for p in embedded]
    print(f"\n  (For reference, RAW pre-centering norms average "
          f"{np.mean(raw_norms):.4f}, so post-centering norms this much "
          f"smaller than that are the ones to worry about.)")

    # ── Step 3: embed the 20 queries and center them the same way ──
    print("\n" + "-" * 70)
    print("QUERY-SIDE NORMS (centered) — checking if queries ALSO collapse")
    print("-" * 70)

    tulu_queries = load_tulu_queries()
    kannada_script = load_kannada_script_tulu()

    query_norms_romanized = []
    for q in tulu_queries:
        vec = get_embedding(q["tulu"]) - corpus_mean
        query_norms_romanized.append(l2_norm(vec))
    print(f"  Romanized Tulu queries — mean centered norm: "
          f"{np.mean(query_norms_romanized):.4f} "
          f"(range {min(query_norms_romanized):.4f}-{max(query_norms_romanized):.4f})")

    if kannada_script:
        query_norms_kannada = []
        for qid, text in kannada_script.items():
            vec = get_embedding(text) - corpus_mean
            query_norms_kannada.append(l2_norm(vec))
        print(f"  Kannada-script Tulu queries — mean centered norm: "
              f"{np.mean(query_norms_kannada):.4f} "
              f"(range {min(query_norms_kannada):.4f}-{max(query_norms_kannada):.4f})")

    # ── Step 4: raw vs. centered similarity for the known hub doc ──
    print("\n" + "-" * 70)
    print("RAW vs CENTERED cosine similarity: does centering INFLATE the hub?")
    print("-" * 70)
    hub_name_guess = min(doc_avg, key=lambda d: doc_avg[d])
    print(f"  Testing against smallest-norm doc found above: {hub_name_guess}")

    hub_chunks_raw = [p for p in embedded if p["case_name"] == hub_name_guess]
    hub_chunks_centered = [p for p in centered if p["case_name"] == hub_name_guess]

    sample_queries = tulu_queries[:5]
    for q in sample_queries:
        raw_q_vec = get_embedding(q["tulu"])
        centered_q_vec = raw_q_vec - corpus_mean

        raw_sim = max(cosine_similarity(raw_q_vec, c["embedding"]) for c in hub_chunks_raw)
        centered_sim = max(cosine_similarity(centered_q_vec, c["embedding"]) for c in hub_chunks_centered)

        print(f"  Query {q['id']:>2} ({q['english_gloss'][:40]:40}) "
              f"raw={raw_sim:.4f}  centered={centered_sim:.4f}  "
              f"{'<-- INFLATED' if centered_sim - raw_sim > 0.2 else ''}")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("  - If the hub doc's centered norm is far smaller than others,")
    print("    AND centered similarity is much higher than raw similarity")
    print("    across unrelated queries -> the norm-collapse hypothesis is")
    print("    confirmed: mean-centering is amplifying the hub problem for")
    print("    this embedding model, not fixing it.")
    print("  - If norms look normal and centered ~ raw similarity -> the")
    print("    hub dominance has a different cause and we should look")
    print("    elsewhere (e.g. genuine hubness unrelated to centering, or")
    print("    the query embeddings themselves failing to differentiate).")
    print("=" * 70)


if __name__ == "__main__":
    main()
