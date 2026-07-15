import json
import csv
import re
import os
import numpy as np
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# TULU LEGAL RAG PIPELINE — v7
#
# Five fixes from literature review (MinPrompt_Legal_Handoff.md):
#
#   1. LIST TAXONOMY: replaced CUAD commercial-contract categories
#      with LIST access-to-justice categories purpose-built for
#      everyday lay legal complaints. Hits@3 against both primary
#      and secondary labels (some situations span two categories).
#
#   2. STATELESS CALLS: killed KV-cache continuity across the 20
#      sentences. Each call is now independent with a shared static
#      prefix (category definitions + few-shot examples). Eliminates
#      "lost in the middle" degradation and mode/category collapse.
#      Source: Liu et al. TACL 2024; Holtzman et al. 2019.
#
#   3. EMBEDDING MODEL: default is now multilingual-e5-large-instruct
#      (HuggingFace). Diagnostic testing found nomic-embed-text's
#      Neelamegham dominance was NOT a mean-centering artifact -- raw,
#      uncentered nomic scores (0.97-1.0) were nearly identical to
#      centered ones, and Neelamegham's apparent 40% "accuracy" was a
#      majority-baseline coincidence: its 2 broad category tags
#      (Land/Property, Courts/Legal-System) happened to overlap with
#      8 of 20 query labels regardless of query content -- a perfect
#      20/20 correlation between "hit" and "query touches one of
#      those 2 tags," confirming zero real discrimination. Raw E5
#      matched nomic's aggregate accuracy (40%/75% vs 40%/85%) but
#      through 10 distinct documents winning top-1 instead of 1 --
#      i.e. genuine (if imperfect) topical discrimination rather than
#      a lucky coincidence. Falls back to MuRIL, then nomic-embed-text
#      if E5 is unavailable. See E5_MODEL_NAME / USE_MEAN_CENTERING.
#
#   4. HUB FIXES: three-layer approach to Neelamegham dominance:
#      (a) mean-centering of all chunk embeddings before retrieval --
#          NOW OPTIONAL (USE_MEAN_CENTERING, default False). Diagnosed
#          as not the cause of nomic's collapse and never validated
#          against E5; toggle this to test empirically rather than
#          assume it helps.
#      (b) per-document chunk quota: max 2 chunks per doc in top-k
#      (c) MMR re-ranking (lambda=0.5) for diversity
#      Source: Radovanovic et al. 2010; Carbonell & Goldstein 1998.
#
#   5. QUERY-FIRST PROMPTING: situation placed both BEFORE and AFTER
#      retrieved passages; explicit instruction that passages are
#      background law only — do not classify the passage, classify
#      the situation. Source: Liu et al. TACL 2024; FILCO/RAGAS.
#
# Four conditions, same 20 sentences each time:
#   A. English gloss + RAG        — upper bound
#   B. Tulu (romanized) + RAG     — true low-resource floor
#   C. Tulu + Phase 1 priming + RAG
#   D. Tulu (Kannada script) + Kannada priming + RAG
#      (auto-skips if reviewed Kannada file not present)
#
# Evaluation: Hits@1 (exact match) and Hits@3 (correct label in
# top-3 predictions) against LIST primary + secondary labels.
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KANNADA_CORPUS_PATH = os.path.join(SCRIPT_DIR, "kannada_legal_corpus_list.jsonl")
TULU_DATASET_PATH   = os.path.join(SCRIPT_DIR, "tulu_legal_20_list.jsonl")
TULU_KANNADA_CANDIDATE_PATHS = [
    os.path.join(SCRIPT_DIR, "tulu_legal_20_kannada.jsonl"),
    os.path.join(SCRIPT_DIR, "tulu_legal_20_kannada_DRAFT_needs_review.jsonl"),
]

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
MMR_LAMBDA    = 0.5   # balance relevance vs diversity in MMR
MAX_CHUNKS_PER_DOC = 2  # hub quota: no doc contributes more than this to top-k
TOP_K         = 3

# ── EMBEDDING MODEL CONFIG ──────────────────────────────────────
# E5 diagnostics (raw, no centering/quota/MMR) showed real topical
# discrimination -- 40% Hits@1 / 75% Hits@3 vs ~12.5% random chance,
# with 10 distinct documents winning top-1 across 20 queries (vs
# nomic's 1 document winning 19/20). E5 is now the default.
E5_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
E5_TASK_DESCRIPTION = "Given a legal situation described by a person, retrieve relevant legal case documents"

# Mean-centering was diagnosed as NOT the cause of nomic's hub collapse
# (raw nomic scores were nearly identical to centered ones -- 0.97-1.0
# either way) and was NEVER TESTED against E5. Default is OFF since raw
# E5 already showed real diversity with no centering at all; flip this
# to True to test whether centering helps, hurts, or does nothing for E5
# before committing to either setting for the full paper run.
USE_MEAN_CENTERING = False

# ── LIST CATEGORIES ───────────────────────────────────────────
# Purpose-built for lay legal complaints (Stanford Legal Design Lab)
# replacing CUAD's commercial-contract clause taxonomy.

LIST_CATEGORIES = [
    "Housing",
    "Consumer/Money/Debt",
    "Work/Employment",
    "Land/Property",
    "Torts/Individuals",
    "Courts/Legal-System",
    "IP/Identity",
    "Estates/Wills",
]

LIST_DEFINITIONS = {
    "Housing": (
        "landlord-tenant disputes, security deposits not returned, trespass into "
        "a home or dwelling, damage to rented property, eviction, someone entering "
        "your house without permission. Examples: 'My landlord did not give back my "
        "deposit', 'Someone came into my house without asking'."
    ),
    "Consumer/Money/Debt": (
        "money owed but not paid, loans not repaid, overcharging beyond what was "
        "agreed, contract breaches involving payment, fraud where money was taken "
        "and the person disappeared. Examples: 'I lent money and they won't return "
        "it', 'They charged more than we agreed', 'I gave someone money and they "
        "disappeared', 'They ended our agreement early'."
    ),
    "Work/Employment": (
        "wages not paid, wrongful dismissal without reason, employer stopping you "
        "from starting your own business, non-compete situations. Examples: 'My "
        "employer did not pay me this month', 'My boss told me to leave without "
        "any reason', 'My old employer is stopping me from starting my own work'."
    ),
    "Land/Property": (
        "neighbor taking your land, someone occupying your land without permission "
        "and refusing to leave, boundary disputes, land stolen after inheritance. "
        "Examples: 'My neighbor took my land', 'Someone is living on my land "
        "without asking and won't leave'."
    ),
    "Torts/Individuals": (
        "personal injury caused by another person, property damage caused by "
        "another person's negligence, someone breaking your property and not "
        "fixing it, a neighbor's tree falling on your house. Examples: 'Someone "
        "hit me with a stick and my hand hurts', 'Someone broke my fence and did "
        "not fix it', 'My neighbor's tree fell on my house'."
    ),
    "Courts/Legal-System": (
        "confusion about which court to go to, not knowing where to file a "
        "complaint, the other party being in a different location or jurisdiction, "
        "signing a document without understanding what it said. Examples: 'I don't "
        "know which court I should go to', 'The person who owes me money is in a "
        "different place', 'I signed a paper but did not understand what it said'."
    ),
    "IP/Identity": (
        "someone using your name without permission, someone selling things under "
        "your name or brand, misuse of personal or business identity. Examples: "
        "'Someone is selling their things using my name'."
    ),
    "Estates/Wills": (
        "inheritance disputes after a family member's death, family members taking "
        "property that was meant to be given to you, land or assets disputed after "
        "someone passes away. Examples: 'After my father passed away my family "
        "took the land that was meant for me'."
    ),
}

# Few-shot examples reused as static prefix in every call (stateless design).
# Two examples per call — one clear hit, one that required ruling out alternatives.
FEW_SHOT_EXAMPLES = """Here are two example classifications to calibrate your reasoning:

Example 1:
Situation: "My landlord kept my deposit money after I moved out."
Classification reasoning: The person had a rental relationship (Housing) and the
landlord failed to return money (Consumer/Money/Debt). The primary issue is the
landlord-tenant relationship and deposit — Housing.
{"category": "Housing", "confidence": 0.85, "reasoning": "Landlord-tenant deposit dispute"}

Example 2:
Situation: "Someone broke my fence and refused to fix it."
Classification reasoning: This is damage caused by another individual to my property.
It is not a housing/landlord issue (no tenancy). It is not a debt (no money owed from
an agreement). It is physical harm/damage caused by another person — Torts/Individuals.
{"category": "Torts/Individuals", "confidence": 0.80, "reasoning": "Property damage caused by another person's action"}
"""

# ══════════════════════════════════════════════════════════════
# PHASE 1 PRIMING BLOCKS (Condition C — romanized, Condition D — Kannada script)
# ══════════════════════════════════════════════════════════════

PHASE1_PRIMING_BLOCK = """Vocabulary guide for Tulu (romanized):
NOUNS: Anjo=man, Al=she, Balae=child, Ponjo=woman, Nayi=dog, Guru=teacher,
Appae=mother/parent, Neer=water, Meen=fish, Pustaka=book
LEGAL NOUNS: yejamaana=landlord/owner, dhani=employer, kas=money, jagga=land,
ille=house, bailee=fence, nerakare=neighbor, pudar=name/signature, kakaji=document,
kotumba=family, popae=father, sala=loan/debt, oppigay=permission
VERBS: Popuni=to go, Maltuni=to do/make, Korpuni=to give, Gothu=to know,
Bodu=to want, Nenpuni=to think/remember
LEGAL VERBS: detae=took, wapas=return/back, mugi=to end/finish, ontaiyare=to stop/prevent
GRAMMATICAL MARKERS: Ijji=negation(not/none), Att=negative marker, Boca=after,
Dumbu=before, Matra=but/only(contrast), Panda=quotative(THAT), Undu=exists/is
GRAMMAR: SOV word order (verb at end). Negation: sentences end with ijji or att.
Suffix -g/-k = 'to/towards'. Suffix -na/-da = 'of/belonging to'.
Panda marks embedded speech: [what was said] + panda + [reporting verb]."""

PHASE1_PRIMING_BLOCK_KANNADA = """ಈ ಭಾಷೆಗೆ ಪದಕೋಶ ಮಾರ್ಗದರ್ಶಿ (Vocabulary guide for Tulu in Kannada script):
ನಾಮಪದಗಳು: ಆಂಜೊ=man, ಆಲ್=she, ಬಾಲೆ=child, ಅಪ್ಪೆ=mother/parent, ಮೀನ್=fish
ಕಾನೂನು ನಾಮಪದಗಳು: ಯಜಮಾನ=landlord/owner, ಧನಿ=employer, ಕಾಸು=money,
ಜಾಗ=land, ಇಲ್ಲ್=house, ಬೈಲಿ=fence, ನೆರೆಕರೆ=neighbor, ಪುದರ್=name/signature,
ಕಾಗಜಿ=document, ಕುಟುಂಬ=family, ಸಾಲ=loan/debt, ಒಪ್ಪಿಗೆ=permission
ಕ್ರಿಯಾಪದಗಳು: ಪೋಪಿನಿ=to go, ಮಲ್ತುನಿ=to do/make, ಕೊರ್ಪುನಿ=to give,
ಗೊತ್ತು=to know, ಬೋಡು=to want, ವಾಪಸ್=return/back, ಮುಗಿ=to end/finish
ವ್ಯಾಕರಣ: ಇಜ್ಜಿ=negation(not), ಅತ್ತ್=negative, ಬೊಕ್ಕ=after, ದುಂಬು=before,
ಮಾತ್ರ=but/contrast, ಪಂಡ=quotative(THAT), ಉಂಡು=exists/is
SOV word order — verb always at end. Suffix -ಗ್/-ಕ್ = 'to/towards'.
Suffix -ನ/-ದ = 'of/belonging to'. ಪಂಡ marks embedded speech."""

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_tulu_queries(path=TULU_DATASET_PATH):
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            romanized = item.get("tulu", item.get("tulu_latin"))
            if romanized is None:
                raise KeyError(f"Entry id={item.get('id')} missing 'tulu'/'tulu_latin'")
            queries.append({
                "id":           item["id"],
                "tulu":         romanized,
                "english_gloss": item["english_gloss"],
                "list_primary":  item.get("list_category", item.get("cuad_category")),
                "list_secondary": item.get("list_category_secondary", ""),
            })
    print(f"Loaded {len(queries)} queries — LIST taxonomy")
    return queries


def load_kannada_script_tulu(paths=TULU_KANNADA_CANDIDATE_PATHS):
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        print("\n[Condition D skipped] No Kannada-script file found.")
        return None
    print(f"  (Condition D source: {os.path.basename(path)})")
    entries, unreviewed = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            eid = item["id"]
            if item.get("tulu_kannada", "").strip():
                entries[eid] = item["tulu_kannada"].strip()
            elif item.get("tulu_kannada_DRAFT", "").strip():
                if item.get("reviewed_by_native_speaker") is True:
                    entries[eid] = item["tulu_kannada_DRAFT"].strip()
                else:
                    unreviewed.append(eid)
    missing = [i for i in range(1, 21) if i not in entries and i not in unreviewed]
    if unreviewed or missing:
        print(f"\n[Condition D skipped] Not ready — unreviewed: {sorted(unreviewed)}, missing: {sorted(missing)}")
        return None
    print(f"\n[Condition D enabled] {len(entries)} reviewed Kannada-script sentences loaded.")
    return entries


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_kannada_corpus(path=KANNADA_CORPUS_PATH):
    passages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            for i, chunk in enumerate(chunk_text(doc["text"])):
                passages.append({
                    "text":           chunk,
                    "case_name":      doc.get("case_name", doc.get("id")),
                    "doc_id":         doc.get("id"),
                    "chunk_index":    i,
                    "categories":     doc.get("cuad_categories", []),
                    "list_categories": doc.get("list_categories", []),
                })
    n_docs = len(set(p["doc_id"] for p in passages))
    print(f"Loaded {len(passages)} chunks from {n_docs} source documents")
    return passages

# ══════════════════════════════════════════════════════════════
# EMBEDDINGS — MuRIL with nomic-embed-text fallback
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# EMBEDDINGS — MuRIL first (HuggingFace), then Ollama fallbacks
#
# Priority order:
#   1. MuRIL (google/muril-base-cased) via HuggingFace
#      Best for Kannada + romanized Indic text (Khanuja et al. 2021)
#      Requires: pip install transformers torch sentence-transformers
#   2. nomic-embed-text-v2 via Ollama
#      Nomic's newer multilingual model (~100 languages incl. Kannada)
#      Requires: ollama pull nomic-embed-text-v2
#   3. nomic-embed-text v1 via Ollama
#      English-primary fallback, always available
#
# The script detects which is available and uses the best one.
# The embedding model used is printed at startup and saved in the
# CSV filename so results from different models are never mixed up.
# ══════════════════════════════════════════════════════════════

_embed_backend    = None   # "huggingface" or "ollama"
_embed_model_name = None
_hf_tokenizer     = None   # HuggingFace tokenizer if MuRIL loaded
_hf_model         = None   # HuggingFace model if MuRIL loaded


def _try_load_muril():
    """
    Load MuRIL via HuggingFace transformers directly.
    Avoids sentence_transformers which crashes on Windows due to
    symlink limitations in the HuggingFace cache system.
    MuRIL (google/muril-base-cased) is Google's Indic-specific model,
    trained on Kannada + romanized Indic text (Khanuja et al. 2021).
    """
    global _hf_tokenizer, _hf_model, _embed_backend, _embed_model_name
    try:
        print("  Trying MuRIL (google/muril-base-cased) via HuggingFace...", flush=True)
        import torch
        from transformers import AutoTokenizer, AutoModel

        _hf_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
        _hf_model     = AutoModel.from_pretrained("google/muril-base-cased")
        _hf_model.eval()

        # Quick test to confirm it produces embeddings
        inputs  = _hf_tokenizer("test", return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = _hf_model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        _embed_backend    = "huggingface"
        _embed_model_name = "muril-base-cased"
        print(f"  Embedding model: MuRIL (HuggingFace transformers) — dim: {len(vec)}", flush=True)
        return True
    except ImportError as e:
        print(f"  [skip] Missing dependency: {e}", flush=True)
        print("         pip install transformers torch", flush=True)
        return False
    except BaseException as e:
        print(f"  [skip] MuRIL failed: {type(e).__name__}: {e}", flush=True)
        return False


def _try_load_e5():
    """
    Load multilingual-e5-large-instruct via HuggingFace transformers.
    Diagnostic testing (raw retrieval accuracy, no pipeline fixes applied)
    showed 40% Hits@1 / 75% Hits@3 against ~12.5% random chance, with
    top-1 wins spread across 10 distinct corpus documents -- versus
    nomic-embed-text's near-total collapse onto a single document
    (Neelamegham, 19/20 queries) whose "accuracy" turned out to be a
    majority-baseline artifact (its 2 broad category tags happened to
    overlap with 8 of 20 query labels by chance), not real retrieval.
    """
    global _hf_tokenizer, _hf_model, _embed_backend, _embed_model_name
    try:
        print(f"  Trying {E5_MODEL_NAME} via HuggingFace...", flush=True)
        import torch
        from transformers import AutoTokenizer, AutoModel

        _hf_tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
        _hf_model     = AutoModel.from_pretrained(E5_MODEL_NAME)
        _hf_model.eval()

        # Quick test to confirm it produces embeddings
        inputs  = _hf_tokenizer("test", return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = _hf_model(**inputs)
        dim = outputs.last_hidden_state.shape[-1]

        _embed_backend    = "huggingface_e5"
        _embed_model_name = "multilingual-e5-large-instruct"
        print(f"  Embedding model: E5-large-instruct (HuggingFace transformers) — dim: {dim}", flush=True)
        return True
    except ImportError as e:
        print(f"  [skip] Missing dependency: {e}", flush=True)
        print("         pip install transformers torch sentencepiece", flush=True)
        return False
    except BaseException as e:
        print(f"  [skip] E5 failed: {type(e).__name__}: {e}", flush=True)
        return False


def _try_load_ollama_embed(model_name):
    global _embed_backend, _embed_model_name
    try:
        import ollama
        print(f"  Trying {model_name} via Ollama...", flush=True)
        ollama.embed(model=model_name, input="test")
        _embed_backend = "ollama"
        _embed_model_name = model_name
        print(f"  Embedding model: {model_name} (Ollama)", flush=True)
        return True
    except ImportError:
        print("  [skip] ollama package not installed: pip install ollama", flush=True)
        return False
    except BaseException as e:
        print(f"  [skip] {model_name} not available: {type(e).__name__}: {e}", flush=True)
        return False


def _init_embed_model():
    global _embed_backend, _embed_model_name
    if _embed_model_name is not None:
        return
    print("\nInitializing embedding model (trying best available)...", flush=True)
    if _try_load_e5():
        return
    #if _try_load_muril():
        #return
    if _try_load_ollama_embed("nomic-embed-text-v2"):
        return
    if _try_load_ollama_embed("nomic-embed-text"):
        print("  WARNING: nomic-embed-text v1 is English-primary.", flush=True)
        return
    print("\n" + "="*60, flush=True)
    print("ERROR: No embedding model found. Try one of:", flush=True)
    print("  pip install sentence-transformers torch", flush=True)
    print("  ollama pull nomic-embed-text-v2", flush=True)
    print("  ollama pull nomic-embed-text", flush=True)
    print("Also make sure Ollama is running: ollama serve", flush=True)
    print("="*60, flush=True)
    raise RuntimeError("No embedding model available.")


def _muril_embed(text):
    """
    Embed text using MuRIL via HuggingFace transformers.
    Uses CLS token (first token of last hidden state), L2-normalized.
    """
    import torch
    inputs = _hf_tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        outputs = _hf_model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


def _e5_embed(text, is_query=False):
    """
    Embed text using E5-large-instruct via HuggingFace transformers.
    Unlike MuRIL (CLS token), E5 wants MEAN pooling over tokens, and its
    convention is: queries get an instruction prefix, passages/documents
    do not. Getting either of these wrong measurably hurts differentiation
    (confirmed during diagnostics -- with-prefix pairwise query similarity
    was higher/worse than without).
    """
    import torch
    if is_query:
        text = f"Instruct: {E5_TASK_DESCRIPTION}\nQuery: {text}"

    inputs = _hf_tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        outputs = _hf_model(**inputs)

    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = (summed / counts).squeeze().numpy()

    norm = np.linalg.norm(mean_pooled)
    return (mean_pooled / norm).astype(np.float32) if norm > 0 else mean_pooled.astype(np.float32)


def get_embedding(text, is_query=False):
    """
    Embed text using the best available model.
    Always returns a normalized numpy float32 array.

    is_query: True for retrieval queries, False for corpus passages.
    Only matters for E5 (instruction-prefix convention) -- MuRIL and
    nomic ignore it.
    """
    _init_embed_model()

    if _embed_backend == "huggingface_e5":
        return _e5_embed(text, is_query=is_query)
    elif _embed_backend == "huggingface":
        return _muril_embed(text)
    else:
        import ollama
        try:
            r = ollama.embed(model=_embed_model_name, input=text)
            return np.array(r["embeddings"][0], dtype=np.float32)
        except Exception:
            r = ollama.embeddings(model=_embed_model_name, prompt=text)
            return np.array(r["embedding"], dtype=np.float32)


def embed_corpus(passages):
    _init_embed_model()
    print(f"\nEmbedding {len(passages)} passages with {_embed_model_name}...")
    embedded = []
    for i, p in enumerate(passages):
        try:
            vec = get_embedding(p["text"])
            embedded.append({**p, "embedding": vec})
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(passages)}")
        except Exception as e:
            print(f"  Error on passage {i}: {e}")
    print(f"Embedded {len(embedded)} passages")
    return embedded


def mean_center(embedded):
    """
    Mean-centering: subtract corpus mean embedding from all chunk vectors.
    Standard hubness remedy. Suppresses hub documents (e.g. Neelamegham)
    whose embeddings sit close to the corpus centroid.
    """
    vecs = np.stack([p["embedding"] for p in embedded])
    mean = vecs.mean(axis=0)
    centered = []
    for p in embedded:
        centered.append({**p, "embedding": p["embedding"] - mean})
    print(f"  Mean-centered {len(centered)} embeddings (hub suppression)")
    return centered, mean

# ══════════════════════════════════════════════════════════════
# RETRIEVAL — cosine + hub quota + MMR
# ══════════════════════════════════════════════════════════════

def cosine_similarity(v1, v2):
    dot  = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0


def retrieve(query_text, embedded_corpus, corpus_mean,
             top_k=TOP_K, max_per_doc=MAX_CHUNKS_PER_DOC, mmr_lambda=MMR_LAMBDA):
    """
    Three-layer hub fix:
      (a) Mean-center query using same corpus mean (applied at embed time above)
      (b) Per-document chunk quota: each doc contributes at most max_per_doc chunks
      (c) MMR re-ranking for diversity within top candidates
    """
    query_vec = get_embedding(query_text, is_query=True) - corpus_mean

    # Score all passages
    scored = [
        {**p, "score": cosine_similarity(query_vec, p["embedding"])}
        for p in embedded_corpus
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Per-document quota (fix 4b)
    quota_filtered = []
    doc_counts = {}
    for p in scored:
        doc_id = p["doc_id"]
        doc_counts[doc_id] = doc_counts.get(doc_id, 0)
        if doc_counts[doc_id] < max_per_doc:
            quota_filtered.append(p)
            doc_counts[doc_id] += 1
        if len(quota_filtered) >= top_k * 5:  # candidate pool for MMR
            break

    # MMR re-ranking (fix 4c)
    selected = []
    candidates = quota_filtered[:]
    while len(selected) < top_k and candidates:
        if not selected:
            # First pick: highest relevance
            best = max(candidates, key=lambda x: x["score"])
        else:
            # MMR: balance relevance vs redundancy with already-selected
            def mmr_score(c):
                rel = c["score"]
                red = max(cosine_similarity(c["embedding"], s["embedding"])
                          for s in selected)
                return mmr_lambda * rel - (1 - mmr_lambda) * red
            best = max(candidates, key=mmr_score)
        selected.append(best)
        candidates.remove(best)

    return selected

# ══════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION — query-first, stateless, LIST taxonomy
# ══════════════════════════════════════════════════════════════

def category_list_block():
    return "\n".join(
        f"- {c}: {LIST_DEFINITIONS[c]}" for c in LIST_CATEGORIES
    )


def format_retrieved_context(retrieved):
    ctx = ""
    for i, p in enumerate(retrieved):
        # Prefer LIST categories if available, fall back to CUAD
        labels = p.get("list_categories") or p.get("categories", [])
        label_str = ", ".join(labels) if labels else "not annotated"
        ctx += (
            f"\nPassage {i+1} (similarity: {p['score']:.3f}, "
            f"source: {p.get('case_name','?')}, "
            f"LIST categories for this case: {label_str}):\n"
            f"{p['text']}\n"
        )
    return ctx


def build_static_prefix(include_priming=False, kannada_script_mode=False):
    """
    Shared static prefix sent identically in every call for this condition.
    Contains: system role + LIST category definitions + anti-collapse warning
    + few-shot examples + optional priming.
    This replaces the accumulating KV-cache context from v4.
    """
    priming = ""
    if include_priming:
        priming = (
            f"\n{PHASE1_PRIMING_BLOCK_KANNADA}\n\n"
            if kannada_script_mode
            else f"\n{PHASE1_PRIMING_BLOCK}\n\n"
        )

    return (
        "You are a legal AI assistant helping classify everyday legal situations "
        "described by lay speakers into legal issue categories.\n\n"
        "The legal categories are:\n"
        f"{category_list_block()}\n\n"
        "IMPORTANT: These categories describe EVERYDAY PROBLEMS, not commercial "
        "contract clauses. Classify based on what the PERSON'S SITUATION is about, "
        "NOT based on legal terminology in retrieved passages.\n\n"
        "Do not default to the same category repeatedly. Weigh each situation "
        "against ALL categories before deciding.\n\n"
        f"{FEW_SHOT_EXAMPLES}"
        f"{priming}"
    )


def build_call_prompt(static_prefix, display_text, english_gloss,
                       retrieved, script_label, kannada_script_mode=False,
                       no_rag=False):
    """
    Query-first prompting. If no_rag=True, skips retrieved passages entirely
    and classifies from situation + priming only. Used for ablation Condition E
    (no-RAG baseline) to test whether priming alone drives Condition D's result.
    """
    if english_gloss:
        situation = (
            f"SITUATION (in {script_label}): \"{display_text}\"\n"
            f"TRANSLATION: \"{english_gloss}\""
        )
    elif kannada_script_mode:
        situation = (
            f"SITUATION (Tulu written in Kannada script): \"{display_text}\"\n"
            f"No translation available. Use the vocabulary guide above and any "
            f"Kannada words you recognize in the query to understand the situation."
        )
    else:
        situation = (
            f"SITUATION (Tulu, romanized): \"{display_text}\"\n"
            f"No translation available. Use any patterns you can detect in the text."
        )

    sentence_instructions = (
        "Step 1: Re-read the SITUATION above. In one sentence, describe in plain "
        "English what problem this person is facing — ignore the passages for now.\n"
        "Step 2: Based on YOUR description of the situation (not the passages), "
        "identify the best-fitting LIST category and explain why. Rule out at least "
        "two other categories explicitly.\n"
        "Step 3: Provide your TOP 3 category predictions in order, as JSON:\n"
        "```json\n"
        "{\n"
        '  "top1": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."},\n'
        '  "top2": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."},\n'
        '  "top3": {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}\n'
        "}\n"
        "```\n"
        "CRITICAL: Every category value MUST be exactly one of:\n"
        + "\n".join(f"  - {c}" for c in LIST_CATEGORIES)
        + "\nDo NOT invent new category names or use terms from the passages."
    )

    if no_rag:
        # No retrieved passages — classify from situation + priming only
        return (
            f"{static_prefix}"
            f"━━━ CLASSIFY THIS SITUATION ━━━\n\n"
            f"{situation}\n\n"
            f"No retrieved passages are provided for this classification.\n"
            f"Classify based solely on the situation description and the "
            f"category definitions above.\n\n"
            f"REMINDER — classify this situation:\n{situation}\n\n"
            f"{sentence_instructions}"
        )

    ctx = format_retrieved_context(retrieved)
    return (
        f"{static_prefix}"
        f"━━━ CLASSIFY THIS SITUATION ━━━\n\n"
        f"{situation}\n\n"
        f"The passages below are BACKGROUND LAW ONLY. Do NOT classify what the "
        f"passage is about — classify what the PERSON'S SITUATION is about.\n"
        f"Retrieved passages from the Kannada court corpus:\n{ctx}\n\n"
        f"REMINDER — classify this situation:\n{situation}\n\n"
        f"{sentence_instructions}"
    )

# ══════════════════════════════════════════════════════════════
# EXTRACTION + VALIDATION
# ══════════════════════════════════════════════════════════════

def extract_top3(text):
    """Extract top-3 predictions from model response."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            results = []
            for key in ["top1", "top2", "top3"]:
                if key in data:
                    results.append({
                        "category":  data[key].get("category", "ERROR"),
                        "confidence": data[key].get("confidence", 0.0),
                        "reasoning":  data[key].get("reasoning", ""),
                    })
            if results:
                return results
        except json.JSONDecodeError:
            pass

    # Fallback: find any JSON objects with "category"
    matches = re.findall(r'\{[^{}]*"category"[^{}]*\}', text, re.DOTALL)
    results = []
    for m in matches[:3]:
        try:
            results.append(json.loads(m))
        except json.JSONDecodeError:
            pass
    if results:
        return results

    return [{"category": "ERROR", "confidence": 0.0, "reasoning": "parse error"}]


def validate_categories(top3):
    """Ensure all predicted categories are in LIST_CATEGORIES."""
    validated = []
    for pred in top3:
        cat = pred.get("category", "ERROR")
        if cat not in LIST_CATEGORIES:
            # Find closest match by word overlap
            best = min(
                LIST_CATEGORIES,
                key=lambda c: -sum(w in cat.lower() for w in c.lower().split("/"))
            )
            pred = {**pred, "category": best,
                    "reasoning": f"[corrected from '{cat}'] " + pred.get("reasoning", "")}
        validated.append(pred)
    # Pad to 3 if needed
    while len(validated) < 3:
        validated.append({"category": LIST_CATEGORIES[0], "confidence": 0.0,
                           "reasoning": "padding"})
    return validated[:3]


def score_hits(top3_categories, primary, secondary):
    """
    Hits@1: top prediction matches primary label.
    Hits@3: any of top-3 matches primary OR secondary label.
    Secondary label gets partial credit (0.5) if it appears in top-3
    but primary does not.
    """
    valid_labels = {primary}
    if secondary:
        valid_labels.add(secondary)

    hits1 = top3_categories[0] == primary
    hits3 = any(c in valid_labels for c in top3_categories)
    # Partial: secondary in top3 but primary not
    partial = (not hits1) and hits3 and (top3_categories[0] != primary)
    return hits1, hits3, partial

# ══════════════════════════════════════════════════════════════
# MAIN CONDITION RUNNER
# ══════════════════════════════════════════════════════════════

def run_condition(condition_name, label, tulu_queries, embedded_corpus, corpus_mean,
                   get_display_text, get_retrieval_query, script_label,
                   include_priming, kannada_script_mode, writer, output_file,
                   no_rag=False):
    import ollama

    print(f"\n{'━'*65}")
    print(f"RUNNING: {label}")
    if no_rag:
        print(f"  [NO-RAG MODE: classifying from situation + priming only]")
    print(f"{'━'*65}")

    static_prefix = build_static_prefix(include_priming, kannada_script_mode)

    hits1_count = hits3_count = partial_count = 0

    for query in tulu_queries:
        display_text    = get_display_text(query)
        retrieval_query = get_retrieval_query(query)
        gloss           = query["english_gloss"] if condition_name == "A_WithGloss" else None

        print(f"\n  [{query['id']}] {display_text[:60]}")
        print(f"  English: {query['english_gloss']}")
        print(f"  Expected: {query['list_primary']}"
              + (f" / {query['list_secondary']}" if query['list_secondary'] else ""))

        if no_rag:
            retrieved = []
            print(f"  [No retrieval]")
            doc_names = []
        else:
            retrieved = retrieve(retrieval_query, embedded_corpus, corpus_mean)
            print(f"  Top passage: {retrieved[0]['score']:.3f} ({retrieved[0].get('case_name','?')})")
            doc_names = [r.get('case_name','?').split(' vs')[0][:20] for r in retrieved]
            print(f"  Retrieved docs: {doc_names}")

        prompt = build_call_prompt(
            static_prefix, display_text, gloss, retrieved,
            script_label, kannada_script_mode, no_rag=no_rag
        )

        top3 = [{"category": "ERROR", "confidence": 0.0, "reasoning": "generation failed"}]
        try:
            # FIX 2 — STATELESS: no context= parameter, fresh call every time
            response = ollama.generate(
                model="llama3",
                prompt=prompt,
                options={"temperature": 0.1, "repeat_penalty": 1.1},
            )
            raw  = response["response"].strip()
            top3 = validate_categories(extract_top3(raw))
        except Exception as e:
            print(f"  ERROR: {e}")

        top3_cats = [p["category"] for p in top3]
        hits1, hits3, partial = score_hits(top3_cats, query["list_primary"], query["list_secondary"])

        if hits1:
            hits1_count += 1
        if hits3:
            hits3_count += 1
        if partial:
            partial_count += 1

        h1 = "✓" if hits1 else ("~" if hits3 else "✗")
        print(f"  Top1: {top3_cats[0]} {h1}  Top2: {top3_cats[1]}  Top3: {top3_cats[2]}")
        print(f"  Hits@1: {hits1}  Hits@3: {hits3}")

        writer.writerow([
            condition_name, query["id"], display_text, query["english_gloss"],
            retrieval_query, query["list_primary"], query["list_secondary"],
            top3_cats[0], top3_cats[1], top3_cats[2],
            f"{top3[0]['confidence']:.2f}",
            hits1, hits3, partial,
            f"{retrieved[0]['score']:.4f}" if retrieved else "N/A",
            retrieved[0].get("case_name", "") if retrieved else "N/A",
            " | ".join(doc_names) if doc_names else "N/A",
            top3[0].get("reasoning", ""),
        ])
        output_file.flush()

    n = len(tulu_queries)
    print(f"\n  {label}")
    print(f"  Hits@1: {hits1_count/n*100:.1f}% ({hits1_count}/{n})")
    print(f"  Hits@3: {hits3_count/n*100:.1f}% ({hits3_count}/{n})")
    return {
        "hits1": hits1_count / n * 100,
        "hits3": hits3_count / n * 100,
        "hits1_n": hits1_count,
        "hits3_n": hits3_count,
        "total": n,
    }

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_rag_pipeline():
    import ollama  # noqa — imported here so module loads without ollama present

    print("=" * 65)
    print("TULU LEGAL RAG PIPELINE v7")
    print("LIST taxonomy | Stateless | MuRIL | Hub fixes | Query-first | Hits@3")
    print("=" * 65)

    _init_embed_model()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = _embed_model_name.replace("/", "-").replace(":", "-")
    output_filename = f"tulu_rag_results_v7_{safe_model}_{timestamp}.csv"

    tulu_queries      = load_tulu_queries()
    kannada_script_map = load_kannada_script_tulu()
    corpus            = load_kannada_corpus()
    embedded_corpus   = embed_corpus(corpus)
    if USE_MEAN_CENTERING:
        embedded_corpus, corpus_mean = mean_center(embedded_corpus)
    else:
        corpus_mean = np.zeros_like(embedded_corpus[0]["embedding"])
        print(f"  Mean-centering DISABLED (USE_MEAN_CENTERING=False) — "
              f"using raw embeddings for {len(embedded_corpus)} chunks", flush=True)

    output_file = open(output_filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(output_file)
    writer.writerow([
        "condition", "id", "display_text", "english_gloss",
        "retrieval_query", "list_primary", "list_secondary",
        "pred_top1", "pred_top2", "pred_top3",
        "top1_confidence", "hits1", "hits3", "partial_credit",
        "top_retrieval_score", "top_retrieved_case", "retrieved_doc_names",
        "top1_reasoning",
    ])
    output_file.flush()

    all_results = {}

    all_results["A_WithGloss"] = run_condition(
        "A_WithGloss", "Condition A — English gloss + RAG",
        tulu_queries, embedded_corpus, corpus_mean,
        get_display_text=lambda q: q["tulu"],
        get_retrieval_query=lambda q: q["english_gloss"],
        script_label="Tulu (romanized)",
        include_priming=False, kannada_script_mode=False,
        writer=writer, output_file=output_file,
    )

    all_results["B_TuluOnly"] = run_condition(
        "B_TuluOnly", "Condition B — Tulu (romanized) only + RAG",
        tulu_queries, embedded_corpus, corpus_mean,
        get_display_text=lambda q: q["tulu"],
        get_retrieval_query=lambda q: q["tulu"],
        script_label="Tulu (romanized)",
        include_priming=False, kannada_script_mode=False,
        writer=writer, output_file=output_file,
    )

    all_results["C_Phase1Priming"] = run_condition(
        "C_Phase1Priming", "Condition C — Tulu + Phase 1 priming + RAG",
        tulu_queries, embedded_corpus, corpus_mean,
        get_display_text=lambda q: q["tulu"],
        get_retrieval_query=lambda q: q["tulu"],
        script_label="Tulu (romanized)",
        include_priming=True, kannada_script_mode=False,
        writer=writer, output_file=output_file,
    )

    if kannada_script_map:
        all_results["D_KannadaScript"] = run_condition(
            "D_KannadaScript",
            "Condition D — Tulu (Kannada script) + Kannada priming + RAG",
            tulu_queries, embedded_corpus, corpus_mean,
            get_display_text=lambda q: kannada_script_map[q["id"]],
            get_retrieval_query=lambda q: kannada_script_map[q["id"]],
            script_label="Tulu (Kannada script)",
            include_priming=True, kannada_script_mode=True,
            writer=writer, output_file=output_file,
        )

        # Condition E — NO-RAG ablation: Kannada script + Kannada priming, zero retrieval
        # Tests whether priming alone drives Condition D's result, or whether RAG contributes.
        all_results["E_NoRAG_KannadaPriming"] = run_condition(
            "E_NoRAG_KannadaPriming",
            "Condition E — Tulu (Kannada script) + Kannada priming — NO RAG",
            tulu_queries, embedded_corpus, corpus_mean,
            get_display_text=lambda q: kannada_script_map[q["id"]],
            get_retrieval_query=lambda q: kannada_script_map[q["id"]],
            script_label="Tulu (Kannada script)",
            include_priming=True, kannada_script_mode=True,
            no_rag=True,
            writer=writer, output_file=output_file,
        )

        # Condition F — Script alignment ablation: Kannada script + NO priming + RAG
        # Tests whether script alignment alone (without priming) helps, isolating the
        # two components of Condition D.
        all_results["F_KannadaScript_NoPriming"] = run_condition(
            "F_KannadaScript_NoPriming",
            "Condition F — Tulu (Kannada script) + NO priming + RAG",
            tulu_queries, embedded_corpus, corpus_mean,
            get_display_text=lambda q: kannada_script_map[q["id"]],
            get_retrieval_query=lambda q: kannada_script_map[q["id"]],
            script_label="Tulu (Kannada script)",
            include_priming=False, kannada_script_mode=True,
            writer=writer, output_file=output_file,
        )

    output_file.close()

    print(f"\n{'='*65}")
    print("FINAL COMPARISON")
    print(f"{'='*65}")
    print(f"{'Condition':<25} {'Hits@1':>8} {'Hits@3':>8}")
    print(f"{'-'*45}")
    for name, res in all_results.items():
        print(f"{name:<25} {res['hits1']:>7.1f}% {res['hits3']:>7.1f}%")

    if "A_WithGloss" in all_results and "B_TuluOnly" in all_results:
        a1 = all_results["A_WithGloss"]["hits1"]
        b1 = all_results["B_TuluOnly"]["hits1"]
        a3 = all_results["A_WithGloss"]["hits3"]
        b3 = all_results["B_TuluOnly"]["hits3"]
        print(f"\nLanguage barrier cost Hits@1 (A-B): {a1-b1:.1f} pp")
        print(f"Language barrier cost Hits@3 (A-B): {a3-b3:.1f} pp")

    if "D_KannadaScript" in all_results and "E_NoRAG_KannadaPriming" in all_results:
        d1 = all_results["D_KannadaScript"]["hits1"]
        e1 = all_results["E_NoRAG_KannadaPriming"]["hits1"]
        print(f"\nRAG contribution to Condition D Hits@1 (D-E): {d1-e1:.1f} pp")
        print(f"  (positive = RAG helps; negative = RAG hurts; ~0 = priming drives result)")

    if "D_KannadaScript" in all_results and "F_KannadaScript_NoPriming" in all_results:
        d1 = all_results["D_KannadaScript"]["hits1"]
        f1 = all_results["F_KannadaScript_NoPriming"]["hits1"]
        print(f"Priming contribution to Condition D Hits@1 (D-F): {d1-f1:.1f} pp")
        print(f"  (positive = priming helps; ~0 = script alignment alone drives result)")

    print(f"\nResults saved to: {output_filename}")
    print("=" * 65)


if __name__ == "__main__":
    run_rag_pipeline()
