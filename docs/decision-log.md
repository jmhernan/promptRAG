# promptRAG — Decision Log

*Real-time record of every significant choice, failed approach, and discovered assumption violation.*
*Written as it happens, not reconstructed afterward.*

---

## Entry 001 — Research Doc: Codebase audit gap fixes

**Date:** 2026-04-04
**Phase:** Research

**What was done:**
Reviewed `docs/research.md` against the actual source files. Found 11 gaps between what the
research doc described and what the codebase actually contains.

**Findings:**
1. `create_vdb.py` — MPS device detection missing. Only checks `torch.cuda.is_available()`,
   never `torch.backends.mps.is_available()`. Model silently stays on CPU on Apple Silicon.
2. `create_vdb.py` — FAISS coupling deeper than documented. Uses `Dataset.from_pandas()`,
   `dataset.add_faiss_index()`, and `dataset.get_nearest_examples()` — all HuggingFace `datasets`
   library methods. Switching to ChromaDB is not a drop-in swap; it replaces the entire retrieval
   surface.
3. `prompt_generator.py` — Naming collision in `generate_prompts_for_tweets()` line 48:
   `tweets = tp.clean_text(tweet)` overwrites the outer loop parameter when `clean_tweets=True`.
   Masked because `example_script.py` always passes `clean_tweets=False`.
4. `prompt_generator.py` — `format_retrieved_texts()` bug description in research doc was
   imprecise. The concatenated `formatted_texts` string is correct; what's lost is the raw
   `texts`, `samples['code']`, and `scores` from all but the last iteration.
5. File table listed `data/.gitignore` as "Drop" — file doesn't exist. Root `.gitignore`
   already covers `data/`.
6. `src/__init__.py` was missing from the file table entirely.
7. Dependencies section listed `faiss-cpu` and `pandas` as minimum deps, contradicting
   Decisions 1 (ChromaDB) and 4 (Polars).
8. Chunking discussion was a single sentence. No rationale for fixed-size over alternatives.
9. No mention of `datasets` (HuggingFace) library dependency, which the current code uses.
10. No mention of cross-encoder reranking as a known improvement path.
11. SLM model list (Phi-3, Qwen2.5, Gemma-2, Mistral-v0.3) dated from early 2025 — likely
    have newer versions by implementation time.

**Decision:** All 11 gaps fixed in `docs/research.md`. Research Doc is now accurate against
the actual codebase and internally consistent with its own architectural decisions.

**Impact:** These fixes directly shaped the Plan Doc — especially the FAISS-to-ChromaDB
migration path (finding #2), which determined how `vector_store.py` replaces both
`IndexTextEmbeddings` and `TweetPromptGenerator.query_faiss_index()`.

---

## Entry 002 — Research Doc: File location moved

**Date:** 2026-04-04
**Phase:** Research

**What was done:**
Research doc moved from `.github/research.md` to `docs/research.md` to match the location
specified in `copilot-instructions.md`. Created `docs/` directory.

**Decision:** All project documents (`research.md`, `plan.md`, `decision-log.md`) live in
`docs/`. The `.github/` copy is removed.

---

## Entry 003 — Plan Doc: Phase 1 scope decisions

**Date:** 2026-04-04
**Phase:** Plan

**What was decided:**

| Question | Options Considered | Decision | Rationale |
|---|---|---|---|
| Phase 1 scope | Retrieval only / Full pipeline one dataset / Full all three | **Full pipeline, one dataset** | Proves end-to-end flow before scaling to more datasets |
| First dataset | COVID/Poynter / Wolf Archive / Eviction | **Wolf Archive** | Has the labeled benchmark (pretend-play/conflict); strongest evaluation story. User has access to obtain the data. |
| Config format | YAML / TOML | **YAML** | Already shown in copilot-instructions examples; more readable for ML configs |
| Package structure | Flat `src/` / Proper `src/promptrag/` | **Proper package** | Installable via `pip install -e .`, clean imports |
| Ollama in phase 1 | Yes / No | **No — HuggingFace only** | Decision 5 in research.md. Interface designed so Ollama can be added later. |
| Existing code | Refactor in place / Archive and rebuild | **Archive in branch, rebuild fresh** | Cleaner than incremental refactor; old code preserved in `archive/legacy` branch |
| Testing depth | Smoke / Moderate / Comprehensive | **Moderate** — unit tests for key functions | Covers embeddings, vector store, chunker, evaluation metrics. No CI in phase 1. |

**Blocking items identified:**
- Wolf Archive CSV and gold set files must be obtained before implementation Step 4
- Column names in YAML config marked `TBD` until data files are inspected

**Impact:** Plan Doc written at `docs/plan.md` with 11 implementation steps, code snippets
for each module, and test points.

---

## Entry 004 — Wolf Archive has no labels; dual-dataset strategy

**Date:** 2026-04-04
**Phase:** Plan

**What was discovered:**
The Wolf Archive data (`data/4ChildObservation_MasterFile.csv`) contains 1,678 text
observations but **no theme/behavior labels**. The pretend-play/conflict classifications from
Xu & Hernandez (2025) are not present in this file — they would need to be created from scratch.

**Data inspection results:**
- File: `data/4ChildObservation_MasterFile.csv`, 1,678 rows
- Text column: `text` (word count: min 1, max 1,434, median 306, mean 336)
- ID column: `Index`
- Columns: Index, Page, TargetPerson, Location, Date, Time, Duration (min), present1–12, text
- No label/theme/code columns

The COVID/Poynter dataset (`data/poynter_coded_breon_tab.csv`) has 316 rows with human-coded
misinformation labels (`code` column), tab-separated.

**Decision:** Dual-dataset Phase 1 strategy:
- **COVID/Poynter** — used for retrieval evaluation metrics (labeled data → precision@k,
  recall@k, nDCG, MRR). Proves the evaluation infrastructure works.
- **Wolf Archive** — used for ingestion and generation testing (unlabeled, but exercises the
  full pipeline path). Retrieval quality evaluation deferred to Phase 2 when gold set labels
  are created.
- Wolf Archive pretend-play benchmark moved to Phase 2 scope.

**Minor discrepancy noted:** Research doc (from the published paper) says 1,668 observations;
actual CSV has 1,678: 10-row difference. Likely includes additional observations not
used in the published analysis. Not blocking.

**Impact:** Plan Doc updated — Step 0.2 changed from "obtain data" to "dataset inventory"
(both files already present). Step 11 changed from Wolf Archive benchmark to COVID/Poynter
retrieval eval. Second YAML config (`configs/covid_poynter.yaml`) added. Pipeline `ingest()`
updated to handle tab-separated files.

---

## Entry 005 — Plan Doc review fixes

**Date:** 2026-04-04
**Phase:** Plan

**Issues found during plan review:**

1. **`chunk_size` was ambiguous** — config comments said "words" in one place and had no
   annotation in the other. The chunker uses `text.split()` (word-count based), not tokenizer
   token count. These differ by ~1.3x on average. Both YAML configs now explicitly say
   "word count (not tokens)" to prevent confusion during implementation.

2. **Null text values in COVID dataset** — `story_copy` column may contain nulls. Pipeline
   `ingest()` now filters null rows before processing: `df.filter(pl.col(text_column).is_not_null())`.
   Without this, `None` values would crash `SentenceTransformer.encode()`.

3. **`experiments/` directory not gitignored** — ChromaDB persistence directory and SQLite run
   database are generated at runtime and should not be committed. Added `.gitignore` update
   to Step 1 scaffolding.

**Decision:** All three fixed in `docs/plan.md`. No architectural changes — these are
implementation-level bug prevention.

---

## Entry 006 — Wrong embedding model: base MLM vs. fine-tuned retrieval

**Date:** 2026-04-04
**Phase:** Implementation (caught during testing)

**What happened:**
The vector store test `test_add_and_query` failed — querying "feline sitting" returned
"dogs love to play fetch" instead of "the cat sat on the mat." SentenceTransformers logged:
*"No sentence-transformers model found with name answerdotai/ModernBERT-base. Creating a
new one with mean pooling."*

**Root cause:**
The research doc (Section 4) recommended `answerdotai/ModernBERT-base` as the default
embedding model. This is a **base MLM checkpoint** — a masked language model that has not
been fine-tuned for sentence similarity or retrieval. The BEIR/MTEB scores cited in the
research doc are from fine-tuned versions, not the base checkpoint. Using it directly
produces embeddings that are not optimized for semantic search.

The user's original model (`multi-qa-mpnet-base-dot-v1`) works because it was fine-tuned
via contrastive learning specifically for semantic search (Reimers & Gurevych, 2019). The
research doc correctly described this in Section 2 but failed to apply the same logic
when recommending ModernBERT.

This is exactly the kind of error the RPI methodology is designed to catch — a wrong
assumption in the research doc cascading into broken implementation.

**The correct model:** `nomic-ai/modernbert-embed-base` — Nomic AI fine-tuned ModernBERT-base
for retrieval embeddings using their contrastive training pipeline. MTEB 62.62, BEIR nDCG@10
44.98, 768 dimensions. Works with `SentenceTransformer("nomic-ai/modernbert-embed-base")`
out of the box. Requires `search_query:` / `search_document:` input prefixes.

Also discovered: `freelawproject/modernbert-embed-base_finetune_512` — a legal-domain
fine-tune of the same model, trained on court opinions. Directly relevant for eviction records
dataset (Phase 2). 99.6% triplet accuracy on legal text.

**Decision:** Switch default embedding model from `answerdotai/ModernBERT-base` to
`nomic-ai/modernbert-embed-base`.

**Changes made:**
- `configs/wolf_archive.yaml` — model → `nomic-ai/modernbert-embed-base`
- `configs/covid_poynter.yaml` — model → `nomic-ai/modernbert-embed-base`
- `src/promptrag/embeddings.py` — added `query_prefix`/`document_prefix` params and
  `is_query` flag to `encode()` for asymmetric prefix handling
- `src/promptrag/vector_store.py` — passes `is_query=False` for document encoding,
  `is_query=True` for query encoding
- `docs/research.md` Section 4 — added "Critical distinction" note, updated model table
  to clearly mark base MLM vs. fine-tuned retrieval models, updated recommendation
- Collection naming: `wolf_archive_modernbert_embed`, `covid_modernbert_embed`

**NeoBERT impact:** The base `chandar-lab/NeoBERT` checkpoint has the same problem — it's
a base MLM, not fine-tuned for retrieval. A NeoBERT comparison would require finding or
training a retrieval fine-tune. This narrows the immediate comparison to
`nomic-ai/modernbert-embed-base` vs. `multi-qa-mpnet-base-dot-v1`.

**Lesson:** When a paper reports BEIR/MTEB scores for a model architecture, always verify
whether those scores come from the base checkpoint or a fine-tuned version. The model card
and the paper will specify; the HuggingFace model ID alone can be misleading.

---

## Open TODOs

*Tracked items that are not blocking Phase 1 but need attention. Move to a Decision Log entry
when resolved.*

### Retrieval quality
- [ ] Run baseline comparison: `multi-qa-mpnet-base-dot-v1` vs. `nomic-ai/modernbert-embed-base` on Wolf Archive (same queries, compare scores and ranking)
- [ ] Investigate low absolute similarity scores (0.45–0.47) on Wolf Archive — expected for out-of-domain text? Or query phrasing issue?
- [ ] Experiment with query phrasing — observational register vs. abstract queries

### Wolf Archive labeling (Phase 2 gate)
- [ ] Identify the 24 pretend-play episodes from Xu & Hernandez (2025) in `4ChildObservation_MasterFile.csv` by `Index`
- [ ] Create gold set CSV with episode IDs and expert codings
- [ ] Run pretend-play/conflict benchmark test once gold set exists

### COVID/Poynter eval
- [ ] Run `test_covid_eval.py` — verify retrieval metrics on labeled data
- [ ] Validate that `story_copy` column has no nulls (or confirm null filtering works)

### Infrastructure
- [ ] Add `accelerate` to `pyproject.toml` dependencies (manually installed during testing)
- [ ] Update plan.md to reflect what actually shipped vs. planned
- [ ] Refresh SLM model list (Phi-4, Qwen3, Gemma 3 etc.) before first cross-model comparison

### Rename
- [ ] Rename project to `promptRagEval` — do in a single dedicated commit, not mixed with implementation
- [ ] Order: (1) rename GitHub repo in Settings, (2) `git remote set-url origin` locally, (3) rename local directory, (4) update package name in code, (5) commit + push
- [ ] Update copilot-instructions.md, docs/, README after rename
