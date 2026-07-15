"""
TULU LEGAL — RAW RETRIEVAL QUALITY CHECK (E5)
=================================================
The pairwise-similarity checks so far measured absolute cosine magnitude,
which can be misleading: mean-pooled sentence embeddings are known to be
"anisotropic" -- almost ANY two sentences can show high raw cosine
similarity (0.7-0.9+) purely from embedding-space geometry, regardless of
model quality or content relatedness. A model reading "high everywhere"
doesn't necessarily mean it can't discriminate -- it might just have a
high baseline that a real ranking task doesn't care about.

What actually matters for retrieval: does the CORRECT passage rank above
INCORRECT passages for a given query? This script tests that directly --
no mean-centering, no quota, no MMR, just raw E5 embeddings -- and checks
whether the top-1 and top-3 retrieved chunks' LIST categories match the
query's true category.

Run in the same folder as tulu_legal_rag_v7.py. Reuses the E5 model
downloaded in the previous script.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tulu_legal_rag_v7 import (
    load_kannada_corpus,
    load_tulu_queries,
    load_kannada_script_tulu,
    cosine_similarity,
)

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
TASK_DESCRIPTION = "Given a legal situation described by a person, retrieve relevant legal case documents"

_tokenizer = None
_model = None


def _load_e5():
    global _tokenizer, _model
    print(f"Loading {MODEL_NAME}...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModel
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME)
    _model.eval()
    print("  Loaded.", flush=True)


def e5_embed(text, is_query=True):
    import torch
    if is_query:
        text = f"Instruct: {TASK_DESCRIPTION}\nQuery: {text}"
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / counts
    vec = mean_pooled.squeeze().numpy()
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


def main():
    print("=" * 70)
    print("RAW RETRIEVAL QUALITY CHECK — E5, no centering/quota/MMR")
    print("=" * 70)

    _load_e5()

    passages = load_kannada_corpus()
    print(f"\nEmbedding {len(passages)} corpus chunks (passages, no instruction prefix)...")
    for i, p in enumerate(passages):
        p["embedding"] = e5_embed(p["text"], is_query=False)
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

        query_vec = e5_embed(text, is_query=True)
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
    print(f"RAW retrieval accuracy (E5, no pipeline fixes applied):")
    print(f"  Category Hits@1: {hits1}/20 ({100*hits1/20:.1f}%)")
    print(f"  Category Hits@3: {hits3}/20 ({100*hits3/20:.1f}%)")
    print(f"  Random baseline (8 categories): ~12.5%")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("  - If Hits@1/@3 are well above ~12.5% random chance, E5 IS")
    print("    discriminating real topical signal, even though its raw")
    print("    pairwise-similarity numbers looked uniformly high. The")
    print("    anisotropy caveat was right -- worth building E5 into the")
    print("    full pipeline (with hub suppression re-tuned for it).")
    print("  - If Hits@1/@3 are close to random chance, E5 genuinely isn't")
    print("    discriminating topic for these Kannada-script queries, and")
    print("    that's a real ceiling, not a measurement artifact.")
    print("=" * 70)


if __name__ == "__main__":
    main()
