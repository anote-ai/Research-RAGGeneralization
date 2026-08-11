# MinPrompt-Legal: Cross-Lingual RAG for Tulu Legal Access

Investigating whether large language models can help Tulu speakers — a
Dravidian language with ~2M speakers and no legal corpus of its own —
access legal support by retrieving from an existing Kannada/English legal
corpus. Built for submission to the EMNLP 2026 NLLP Workshop.

**Core finding:** minimal-scaffolding comprehension of Kannada-script Tulu
is surprisingly strong (up to 80% classification accuracy, no fine-tuning) —
but adding retrieval-augmented generation causes a severe accuracy
collapse, driven by genuine confabulation, not just noisy retrieval. Several
mitigation attempts (confidence gating, CRAG-style relevance verification,
citation-forcing) failed to fix this; only a simple disagreement guard
(preferring the no-RAG answer when RAG disagrees) recovered the no-RAG
ceiling.

## Setup

**Requirements:**
```bash
pip install ollama sarvamai numpy --break-system-packages
ollama pull llama3
ollama pull hf.co/prithivMLmods/hex-1-f32-GGUF:Q4_K_M
```

**Sarvam API key** — set as an environment variable, never hardcode it:
```bash
export SARVAM_API_KEY="your-key-here"
```
Get a key at [dashboard.sarvam.ai](https://dashboard.sarvam.ai) (free credits on signup).

**Data files** (in `data/`, loaded automatically — no setup needed):
| File | Contents |
|---|---|
| `kannada_legal_corpus_list.jsonl` | 18-document Kannada legal corpus, with LIST category tags |
| `tulu_legal_20_consolidated.jsonl` / `tulu_legal_60_consolidated.jsonl` | Test sentences in every script (romanized/Kannada/Tamil/Malayalam), plus gold labels |
| `tulu_priming_glossary_trilingual_v2.json` | Vocabulary glossary for linguistic priming (Kannada/Tamil/Malayalam/romanized) |

`pipeline/tulu_rag_engine.py` is the shared engine (embedding, retrieval, prompt utilities) — the other three pipeline scripts import from it directly.

## Running the pipeline

```bash
python3 pipeline/tulu_mega_pipeline.py
```

This runs all 19 experimental condition-tier combinations (A, B, C_sparse,
C_full, D_sparse, D_full, E_sparse, E_full, F, G, H, I, J, K_sparse,
K_full, L_sparse, L_full, M_sparse, M_full) across three models: **Sarvam** (API,
Indic-specialized) → **Hex-1** (open-source, Kannada-specialized, via
Ollama) → **Llama3** (general-purpose, via Ollama). ~1,020 individual
model calls total.

**A pre-flight check runs first**, verifying all required files exist and
are structurally complete *before* the (slow) embedding step starts — so a
missing/broken file is caught in seconds, not after a long wait.

**Every row is saved incrementally.** If a run crashes or a model's API
has an outage partway through, nothing beyond the single row in flight is
lost. To resume, point the relevant entry in `RESUME_FROM` (near the top
of the script) at the partial CSV and rerun — already-completed rows are
skipped automatically. Rows that failed silently (API timeouts, parse
errors) are *not* treated as complete and will be retried, not skipped.

## Project structure

```
pipeline/
  tulu_rag_engine.py            Shared engine -- embedding, retrieval, prompt utilities
  tulu_mega_pipeline.py         Main pipeline -- runs the full A-M matrix x 3 models
  tulu_crag_stage2.py           CRAG-style relevance verification (llama3 only)
  tulu_retrieval_provenance.py  Logs retrieved passages per (condition, query), no LLM calls
data/
  kannada_legal_corpus_list.jsonl            The 18-document retrieval corpus
  tulu_legal_20_consolidated.jsonl           20-sentence test subset, every script
  tulu_legal_60_consolidated.jsonl           Full 60-sentence test set, every script
  tulu_priming_glossary_trilingual_v2.json   Priming vocabulary (Kannada/Tamil/Malayalam/romanized)
outputs/    CSV results from prior runs (created automatically if missing)
paper/      LaTeX source for the NLLP 2026 workshop submission
```

## Known issues / limitations

- **Sarvam API reliability**: occasional intermittent timeouts even with
  a 180s client timeout and 7 retries; appears to be infrastructure-side
  flakiness rather than something fixable via prompt/config changes.
  Use the resume system to backfill failed rows after a run.
- **n=20 test sentences**: several findings in the paper are explicitly
  hedged as directional/suggestive rather than statistically definitive
  at this sample size.
- **Tamil/Malayalam transliterations** were generated via `aksharamukha`
  and have not been independently native-speaker verified.
- **Not a peer-reviewed benchmark**: several supporting claims rely on a
  mix of published academic sources and informal community evaluations —
  see inline citations in `paper/paper.tex` for which is which.

## Related documents

- `paper/paper.tex` — full write-up (LaTeX, ACL format) submitted to the
  EMNLP 2026 NLLP Workshop, including methodology, results, and every
  mitigation attempt that failed.

## Citation

If referencing this work, please cite the associated NLLP 2026 workshop
submission (details to be added upon acceptance).
