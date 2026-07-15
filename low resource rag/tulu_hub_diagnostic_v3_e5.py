"""
TULU LEGAL — E5-LARGE-INSTRUCT QUICK CHECK
=============================================
Cheap pre-check before committing to a full A-H re-run: does
multilingual-e5-large-instruct actually differentiate the 20 Kannada-script
Tulu queries from each other, unlike nomic-embed-text (which averaged
0.9029 pairwise similarity -- i.e. barely differentiated at all)?

Only embeds the 20 queries in each script (40 total) -- no corpus embedding
needed for this check, so it's fast even on CPU.

Reuses load_tulu_queries / load_kannada_script_tulu / cosine_similarity
from tulu_legal_rag_v7.py to stay consistent with the real pipeline's data.

SETUP (one-time):
  pip install transformers torch sentencepiece --break-system-packages
  (first run downloads ~1.1GB of model weights from HuggingFace)

Run in the same folder as tulu_legal_rag_v7.py.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tulu_legal_rag_v7 import (
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
    print(f"Loading {MODEL_NAME} (first run downloads ~1.1GB)...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModel
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME)
    _model.eval()
    print("  Loaded.", flush=True)


def e5_embed(text, is_query=True):
    """
    E5-instruct convention: queries get an instruction prefix, passages/
    documents do not. Uses mean pooling over token embeddings (NOT the
    CLS token -- that's a BERT/MuRIL convention, E5 wants mean pooling),
    then L2-normalizes.
    """
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


def avg_pairwise_sim(vecs_by_id):
    ids = list(vecs_by_id.keys())
    sims = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sims.append(cosine_similarity(vecs_by_id[ids[i]], vecs_by_id[ids[j]]))
    return float(np.mean(sims)), float(np.std(sims))


def main():
    print("=" * 70)
    print("E5-LARGE-INSTRUCT QUICK CHECK — query differentiation only")
    print("=" * 70)

    _load_e5()

    tulu_queries = load_tulu_queries()
    kannada_script = load_kannada_script_tulu()
    if kannada_script is None:
        print("Kannada-script file not found -- cannot run this test.")
        return

    print("\nEmbedding 20 romanized + 20 Kannada-script queries with E5...")
    print("(Round A: WITH instruction prefix, as E5 docs recommend)")
    romanized_vecs_instr = {q["id"]: e5_embed(q["tulu"], is_query=True) for q in tulu_queries}
    kannada_vecs_instr = {qid: e5_embed(text, is_query=True) for qid, text in kannada_script.items()}
    rom_mean_i, rom_std_i = avg_pairwise_sim(romanized_vecs_instr)
    kan_mean_i, kan_std_i = avg_pairwise_sim(kannada_vecs_instr)

    print("(Round B: WITHOUT instruction prefix -- isolating the prefix's effect)")
    romanized_vecs_raw = {q["id"]: e5_embed(q["tulu"], is_query=False) for q in tulu_queries}
    kannada_vecs_raw = {qid: e5_embed(text, is_query=False) for qid, text in kannada_script.items()}
    rom_mean_r, rom_std_r = avg_pairwise_sim(romanized_vecs_raw)
    kan_mean_r, kan_std_r = avg_pairwise_sim(kannada_vecs_raw)

    print("(Round C: ENGLISH GLOSSES -- control for sentence-template similarity)")
    english_vecs_raw = {q["id"]: e5_embed(q["english_gloss"], is_query=False) for q in tulu_queries}
    eng_mean_r, eng_std_r = avg_pairwise_sim(english_vecs_raw)

    print("\n" + "-" * 70)
    print("QUERY-TO-QUERY SIMILARITY — E5-large-instruct")
    print("-" * 70)
    print(f"  {'Condition':<32} {'Romanized':>12} {'Kannada-script':>16} {'English':>10}")
    print(f"  {'WITH instruction prefix':<32} {rom_mean_i:>12.4f} {kan_mean_i:>16.4f} {'':>10}")
    print(f"  {'WITHOUT instruction prefix':<32} {rom_mean_r:>12.4f} {kan_mean_r:>16.4f} {eng_mean_r:>10.4f}")

    print("\n" + "-" * 70)
    print("COMPARISON TO nomic-embed-text (from the prior diagnostic run)")
    print("-" * 70)
    print(f"  {'Model':<32} {'Romanized':>12} {'Kannada-script':>16}")
    print(f"  {'nomic-embed-text':<32} {0.5619:>12.4f} {0.9029:>16.4f}")
    print(f"  {'e5-large-instruct (no prefix)':<32} {rom_mean_r:>12.4f} {kan_mean_r:>16.4f}")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("  - The ENGLISH number is the key control. English is in-distribution")
    print("    for E5 -- if E5 can't differentiate the ENGLISH GLOSSES either,")
    print("    the problem was never about Tulu or Kannada script. It means")
    print("    these 20 short, similarly-templated first-person complaint")
    print("    sentences ('my X did not Y') just look alike to embedding")
    print("    models regardless of language -- a sentence-design finding,")
    print("    not a language-representation finding.")
    print("  - If English differentiates well (drops to ~0.3-0.5, similar to")
    print("    nomic's romanized-Tulu number) but Kannada-script stays stuck")
    print("    near 0.85+, THAT confirms it's genuinely about Tulu/Kannada")
    print("    representation, and trying another embedder is worth it.")
    print("=" * 70)


if __name__ == "__main__":
    main()