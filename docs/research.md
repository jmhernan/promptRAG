# promptRAG — Research Document

*Last updated: April 2026. Incorporates original codebase audit, project reframing conversations,
RPI methodology notes, literature on ModernBERT (Warner et al., 2024) and NeoBERT (Le Breton et al., 2025),
and the published Wolf Archive paper (Xu & Hernandez, 2025) which documents the origin implementation
of this pipeline's core semantic search approach.*

---

## 1. Project Vision

### What this is

promptRAG is a research-team-grade RAG (Retrieval-Augmented Generation) pipeline — clean,
reproducible, and honest about what it does. It started as a COVID misinformation detection
pipeline built on the Poynter dataset and is being refactored from the ground up.

The target audience is small: a research team working with institutional and administrative text.
The quality bar is high: the kind of pipeline you'd be proud to attach to a paper. It is not
optimized for mass adoption or plugin marketplaces.

### What makes this different from a toy pipeline

Two things elevate this work beyond a personal script:

**1. Evaluation as the primary contribution.** RAG evaluation is still fragmented and inconsistent
across the field. Most papers pick one automatic metric (ROUGE, BERTScore) that everyone quietly
knows is inadequate, or they run expensive human eval that nobody can reproduce. This pipeline
treats evaluation as a first-class design concern — not scaffolding. That means separate,
composable measurement of retrieval quality and generation quality, consistent logging across runs,
and infrastructure that makes apples-to-apples comparisons tractable.

**2. SLM-in-RAG evaluation.** The research focus is on understanding the true capabilities of
small language models (Phi, Qwen, Gemma, Mistral-7B, and successors) in retrieval-augmented
settings — not just wrapping GPT-4. Small models fail in qualitatively different ways than large
ones: not just "worse," but structurally different failure modes. Understanding where those
failures are, and whether they stem from retrieval quality or generation quality, is the
publishable contribution.

### Domain focus

The pipeline targets **institutional and administrative text** — the category of messy,
high-stakes documents that organizations generate at scale but rarely make sense of
systematically. This includes court records, public health filings, policy documents, housing
authority records, and field research observations.

Key characteristics of this text type:
- **Structurally messy** — inconsistent formatting, legal boilerplate, domain jargon,
  abbreviations. Chunking and preprocessing decisions matter here in ways they don't for
  clean Wikipedia paragraphs.
- **Precision-oriented retrieval** — queries like "what was the stated reason for this eviction"
  are targeted information extraction tasks, not fuzzy semantic search. This distinction shapes
  how retrieval quality is measured.
- **Achievable ground truth** — subject matter experts (lawyers, librarians, anthropologists,
  public health researchers) can label a small gold set, making rigorous evaluation tractable.
- **Real stakes** — results used to study housing discrimination or public health access require
  trustworthy pipelines.

---

## 2. Concrete Datasets and Evaluation Foundation

The pipeline has three real, labeled datasets that serve as its empirical foundation. Two are
active research projects; one is the original test case.

### Dataset 1: Poynter COVID Misinformation (original)

- Short-form tweet text, labeled for misinformation category
- Original domain for which the codebase was built
- Useful as a baseline and for regression testing during refactor
- Limitation: hardcoded throughout the current codebase (column names `story`, `code`, `rowid`)

### Dataset 2: Eviction Court Records

- Long-form legal text extracted from public court documents
- Task: extract "reasons for eviction" — targeted information extraction from noisy text
- Labeled by subject matter experts; gold set available
- Text regime: long documents, formulaic structure, legal boilerplate
- Directly representative of the institutional text focus

### Dataset 3: Wolf Archive — Children's Behavioral Field Notes

**Published:** Xu, J. & Hernandez, J. M. (2025). "Reading children's moral dramas in anthropological
fieldnotes: A human–AI hybrid approach." *Cambridge Forum on AI: Culture and Society*, 1, e6, 1–21.
doi:10.1017/cfc.2025.10008

This is the origin dataset for the semantic search approach that became promptRAG. The pipeline
in this paper is the direct predecessor of this codebase.

**Corpus:** The Wolf Archive — 1,668 Child Observation (CO) documents collected by anthropologist
Arthur Wolf and a team of local research assistants in rural Taiwan, 1958–60. Naturalistic observations
of children's peer interactions in a Hoklo Han village, averaging ~250 words per document. The first
ethnographic study focused on Han Chinese children.

**Task:** Asymmetric semantic search — six ethnographic themes (family, school, play, cooperation,
conflict, shopping) encoded as fixed "theme vectors"; each observation scored against all six
simultaneously via cosine similarity. Softmax normalization produces scores that sum to 1 per
observation, enabling relative magnitude comparison across themes.

**Implementation (Section 3.3 of the paper):**
- Model: `multi-qa-mpnet-base-dot-v1` — **not plain BERT cosine similarity.** This is a
  sentence-transformers model trained specifically for semantic similarity via contrastive
  learning (Reimers & Gurevych, 2019). Mean pooling on 768-dimensional vectors.
- Index: FAISS with Maximum Inner Product Search (HIPS) and Hierarchical Navigable Small
  World (HNSW) graph for fast retrieval
- Theme vectors constructed from keyword lists derived by integrating topic modeling outputs,
  ethnographic close-reading, and S-BERT word similarity expansion
- Preprocessing: stop word removal, no stemming or lemmatization (preserved negation words,
  pronouns, semantic complexity)

**Gold set:** Anthropologist-assessed labels available for a sample. Manual theme rankings by
domain expert used to validate cosine similarity outputs — relative magnitudes largely matched
human ranking.

**Text regime:** Short, idiosyncratic, observational — structurally the opposite of eviction
documents. Dense with cultural context, meta-communicative cues, and implicit meaning that is
not recoverable from surface lexical features alone.

**Key documented failure mode — directly relevant to eval design:**
S-BERT consistently ranked pretend-play episodes as "conflict" dominant. The model was
misled by high-frequency conflict-associated verbs ("kill," "hit," "die") that appear in
children's playful language. The opening vignette — "Kill that Strange Bird," a Cold War
pretend-play game — was scored as conflict-dominant despite being a cooperative imaginative
game. This failure is not a quirk of one corpus: it reflects a fundamental limitation of
surface-level semantic similarity for tasks requiring meta-communicative interpretation.
The GPT-3.5-turbo follow-up analysis partially improved on this but introduced adult-centric
and Western-centric moral biases of its own.

**Why this matters for promptRAG evaluation design:**
This documented failure is a concrete test case for what the evaluation layer needs to measure.
A retrieval metric that only measures cosine similarity scores would miss this failure entirely.
The gap between "high similarity score" and "actually retrieved the right thing" is exactly
what rigorous evaluation must capture. The pretend-play/conflict confusion is a reproducible,
labeled, domain-expert-validated example of retrieval quality failure that belongs in the
evaluation test suite.

### Why these three matter together

The three datasets span fundamentally different text regimes, retrieval tasks, and failure modes:

| Dataset | Text length | Structure | Task | Known failure mode |
|---|---|---|---|---|
| COVID tweets | Very short | Informal, noisy | Misinformation classification | Baseline / regression |
| Eviction records | Long | Formulaic legal | Targeted extraction | TBD |
| Wolf Archive CO | Short (~250 words) | Observational, implicit | Semantic theme classification | Pretend-play/conflict confusion (documented) |

A pipeline that handles all three honestly — including surfacing the known failure mode on the
Wolf Archive — is making a genuine empirical claim about robustness. The Wolf Archive in
particular provides something rare: a documented, expert-validated retrieval failure with labeled
ground truth. That is the ideal material for developing and validating evaluation methodology.

This breadth separates "I tested on one dataset" from "I characterized where this approach works
and why" — the latter is what a research contribution actually looks like.

---

## 3. Working Methodology: RPI

This project follows the **Research → Plan → Implement (RPI)** methodology. The core principle
is context engineering: never jump to implementation without first compressing truth from the
actual codebase into a research document and a plan.

### Why RPI

LLMs are stateless. The only way to get reliable output from a coding agent is to put reliable
context in. The RPI methodology keeps the context window lean and accurate by front-loading the
thinking work into documents that compress what actually matters.

The failure mode to avoid: a bad line of research (a misunderstanding of how something works) can
cascade into hundreds of bad lines of code. The model will execute your plan; it won't catch your
wrong assumptions.

### Research phase

Research documents must:
- Compress truth from the actual codebase, not from memory or assumptions
- Identify the exact files and line numbers that matter to the problem being solved
- Be reviewed by a human before a plan is written — do not outsource the thinking

### Plan phase

Plans must:
- Reference exact file names and line numbers
- Include code snippets showing what will change (not just prose descriptions)
- Be specific enough that a human reviewer can check whether the approach is correct
- Be scaled to task complexity — a button color change doesn't need a full plan; a module
  refactor does

### Implement phase

- Execute against the plan
- Do not improvise scope during implementation
- Test as specified in the plan

### Context management

Keep context windows lean. When a session goes off track or fills up, use intentional compaction:
ask the agent to compress the current state into a markdown file, review it, and start fresh with
the compressed document rather than the full session history.

---

## 4. Embedding Model Landscape

The current model (`multi-qa-mpnet-base-dot-v1`, 2021) is functional but outdated. The
literature added to this project covers two state-of-the-art encoder-only models that are
directly relevant choices.

### Why encoder-only models still matter for RAG

Despite the rise of LLMs, encoder-only models remain the workhorse of RAG pipelines for a
specific reason: they produce dense vector representations efficiently at scale. As of late 2024,
over half of the 100 most downloaded models on HuggingFace were encoder-based retrieval models.
LLMs depend on encoder-based semantic search as a core component of their retrieval layer.

The problem is that most production RAG pipelines still rely on the original BERT (2019) or
models trained shortly after it, without leveraging the architectural improvements developed since
then.

### ModernBERT (Warner et al., 2024 — Answer.AI / LightOn / JHU)

**arXiv:2412.13663 | Released December 2024**

ModernBERT is the most directly relevant model for this pipeline. It represents a full modernization
of the encoder-only architecture with production RAG as an explicit design target.

**Architecture changes from BERT:**
- **RoPE (Rotary Positional Embeddings)** instead of absolute positional embeddings — better
  length generalization, easier context extension
- **GeGLU activation** (Gated Linear Unit variant) — consistent empirical improvement over GeLU
- **Alternating attention** — every third layer uses global attention (all tokens attend to all);
  remaining layers use 128-token local sliding window attention. Reduces compute on long sequences
  while maintaining global context
- **Unpadding** — removes padding tokens before the embedding layer, concatenates sequences,
  processes as a batch of one. 10-20% throughput improvement over naive unpadding
- **Flash Attention 2/3** — memory-efficient attention kernels; FA3 for global layers, FA2 for
  local layers
- **Pre-normalization** with standard LayerNorm — training stability
- **Modern BPE tokenizer** (modified OLMo tokenizer) — vocabulary size 50,368; better token
  efficiency on code

**Training:**
- 2 trillion tokens of primarily English data: web, code, scientific literature
- Native sequence length: 8,192 tokens (extended from 1,024 via continued training)
- StableAdamW optimizer with trapezoidal LR schedule (WSD)
- Two sizes: base (149M params, 22 layers) and large (395M params, 28 layers)

**Performance highlights (from Table 1 and efficiency benchmarks):**
- State-of-the-art on BEIR retrieval (41.6 base, 44.0 large — beats all existing encoders)
- Best NLU score on GLUE for an MLM-trained model (88.4 base — first to beat DeBERTaV3)
- Dominant on code retrieval (56.4 on CodeSearchNet base, 73.6 on StackQA)
- On long-context ColBERT retrieval (MLDR out-of-domain): 80.2 base, 80.4 large — at least 9
  NDCG@10 points ahead of nearest long-context competitor
- **Inference speed**: on variable-length inputs (typical in practice), processes 14.5-30.9%
  more tokens/sec than GTE-en-MLM at short context and 98.8-118.8% more at long context
- **Memory**: base can process batch sizes 2x larger than any competing model

**Relevance to this project:**
- Best-in-class encoder for RAG retrieval at both short and long context
- Designed for the exact use case: encoder-based semantic search as retrieval layer for LLMs
- Native 8192 context covers the long legal documents in the eviction dataset
- Code-aware tokenizer is not needed here but signals training data breadth
- The alternating local/global attention design may explain strong ColBERT performance —
  relevant if multi-vector retrieval is explored

**HuggingFace model IDs:** `answerdotai/ModernBERT-base`, `answerdotai/ModernBERT-large`

**Critical distinction — base MLM vs. fine-tuned retrieval model:**
The model IDs above are **base MLM checkpoints** — masked language models that have not been
fine-tuned for sentence similarity or retrieval. They cannot be used directly as embedding
models in a RAG pipeline. This is the same distinction as BERT-base vs. `multi-qa-mpnet-base-dot-v1`:
the former is a raw encoder, the latter is fine-tuned via contrastive learning for semantic search.

The correct retrieval-ready model is **`nomic-ai/modernbert-embed-base`** — fine-tuned from
ModernBERT-base by Nomic AI using their contrastive training pipeline (Nussbaum et al., 2024).
MTEB overall 62.62, BEIR nDCG@10 44.98, 768-dimensional vectors, supports Matryoshka
dimensions (256). Requires input prefixes: `search_query:` for queries, `search_document:` for
documents. This is the model that should be used in the pipeline.

Also notable: **`freelawproject/modernbert-embed-base_finetune_512`** — a legal-domain fine-tune
of `modernbert-embed-base` trained on court opinions (Free Law Project). Directly relevant for
the eviction records dataset. 99.6% triplet accuracy on legal text. Phase 2 candidate.

---

### NeoBERT (Le Breton et al., 2025 — Mila / Polytechnique Montréal)

**arXiv:2502.19587 | Released February 2025**

NeoBERT is a concurrent modernization effort from the academic side (Mila), with a different set
of design trade-offs and a stronger emphasis on reproducibility and fair evaluation.

**Architecture changes from BERT:**
- **RoPE** (same motivation as ModernBERT)
- **SwiGLU activation** (vs GeGLU in ModernBERT — both are GLU variants; ablation-informed choice)
- **RMSNorm** instead of LayerNorm — comparable stability, slightly less compute (one fewer
  statistic). This is the choice made by LLaMA family and is more aggressive than ModernBERT's
  LayerNorm
- **Pre-Layer Normalization** — same as ModernBERT
- **Optimal depth-to-width ratio** — 28 layers, hidden size 768 (250M params). BERT and RoBERTa
  were width-inefficient; NeoBERT corrects this while keeping hidden size compatible with
  existing BERT-sized downstream architectures (plug-and-play replacement)
- No alternating attention (all global attention) — simpler, faster, but less efficient on very
  long sequences
- **YaRN-compatible** for further context extension at inference

**Training:**
- RefinedWeb dataset (600B tokens, ~18x larger than RoBERTa's corpus)
- 2.1T tokens seen total (theoretical — sequences padded to max length)
- Two-stage context extension: 1M steps at 1024 tokens, then 50K steps at 4096 tokens
- 20% masking rate (vs BERT's 15%, motivated by Wettig et al. 2023 findings on optimal masking)
- 100% mask replacement (vs BERT's 80/10/10 scheme) — simplified
- AdamW with cosine decay LR schedule

**Performance highlights:**
- Outperforms ModernBERT base **and large** on MTEB under identical fine-tuning conditions
- Does this with 250M params vs 395M for ModernBERT-large
- Faster inference than ModernBERT (no alternating attention overhead, simpler architecture)
- 4096 token context (shorter than ModernBERT's 8192, but covers most institutional documents)

**Relevance to this project:**
- Strong alternative to ModernBERT, especially where inference speed matters
- The ablation-driven design philosophy and fully open release (code, data, checkpoints,
  training scripts) aligns with this project's reproducibility goals
- The standardized MTEB fine-tuning framework NeoBERT introduces is directly applicable here —
  the same impulse motivates the evaluation layer design
- 4096 context may be a practical upper bound for most documents in the eviction and field note
  datasets

**HuggingFace model ID:** `chandar-lab/NeoBERT`

---

### Model Comparison Summary

| Model | Params | Max Context | Key Architecture | MTEB/BEIR | Notes |
|---|---|---|---|---|---|
| `multi-qa-mpnet-base-dot-v1` | 109M | 512 | Original BERT | — | Current; outdated but fine-tuned for retrieval |
| `nomic-ai/modernbert-embed-base` | 149M | 8192 | Modern encoder | 62.6/45.0 | **Recommended default** — fine-tuned for retrieval |
| `BAAI/bge-base-en-v1.5` | 109M | 512 | BERT-based | Strong | Well-benchmarked baseline |
| `all-MiniLM-L6-v2` | 22M | 256 | Distilled | Fast | Lightweight baseline |
| `nomic-ai/nomic-embed-text-v1.5` | 137M | 8192 | NomicBERT | Good | Matryoshka dims |
| `answerdotai/ModernBERT-base` | 149M | 8192 | Modern encoder | — | **Base MLM only** — not for direct retrieval use |
| `answerdotai/ModernBERT-large` | 395M | 8192 | Modern encoder | — | **Base MLM only** — needs fine-tuning |
| `chandar-lab/NeoBERT` | 250M | 4096 | Modern encoder | Best MTEB | **Base MLM only** — needs fine-tuning |
| `freelawproject/modernbert-embed-base_finetune_512` | 149M | 8192 | Modern encoder | — | Legal-domain fine-tune; Phase 2 for eviction |

**Working recommendation:** Start with `nomic-ai/modernbert-embed-base` as the default — it is
ModernBERT fine-tuned for retrieval and works with SentenceTransformers out of the box. Requires
`search_query:` / `search_document:` prefixes. Compare against `multi-qa-mpnet-base-dot-v1`
(the published Wolf Archive model) as the baseline to measure whether the architecture upgrade
improves retrieval quality. NeoBERT comparison requires finding or training a retrieval
fine-tune — the base checkpoint is not directly usable.

---

## 5. LLM Backend Design

The LLM backend must be genuinely pluggable — not "you could theoretically swap it" but actually
clean support for the backends the research team will use.

### Priority backends

**Ollama** — Local inference, dead simple, supports Phi, Qwen, Gemma, Mistral, Llama, and most
open models. Should be the first-class local path.

**HuggingFace `transformers` pipeline** — Full control over model internals, good for
experimentation and for models not yet in Ollama. Required for any fine-tuning or intermediate
layer access.

**vLLM** — For when throughput matters (batch evaluation runs over large datasets).

**OpenAI / Anthropic APIs** — Optional extras for comparison baselines, not the assumed default.

### Prompt format handling

Small models are significantly more sensitive to prompt formatting than large ones. Each model
family has its own chat template: Phi, Qwen, Gemma, Llama, and Mistral all differ. Using the
wrong format degrades performance in ways that are easy to misattribute to model capability.

The correct approach: use the tokenizer's `apply_chat_template()` method for model-specific
formatting. Jinja2 templates as the fallback for custom formats or non-HuggingFace models.

The system prompt and instruction should be configurable and not embedded in template files.
`{tweet}` → `{input_text}` as the generic query placeholder.

---

## 6. Evaluation Design (First-Class Concern)

Evaluation is not scaffolding in this project — it is the point. The pipeline should be designed
so evaluation results are its primary output, not a side effect.

### The two-problem structure of RAG evaluation

RAG evaluation conflates two distinct problems that must be measured separately:

**Retrieval quality:** Did you get the right chunks?
- Precision@k, Recall@k, MRR, NDCG@k
- Measured against the labeled gold sets
- Separates failure modes: is the pipeline failing because of bad embeddings, bad chunking, or
  bad generation?

**Generation quality:** Given those chunks, did the model produce a useful answer?
- Faithfulness / groundedness: does the answer reflect what was retrieved?
- Answer relevance: does the answer address the query?
- Task-specific: for extraction tasks (eviction reasons), exact-match or partial-match against
  labeled spans; for classification tasks (playground behaviors), accuracy/F1 against labels

**End-to-end quality:** Does the full pipeline produce outputs useful for the research task?
This is where SLM evaluation gets interesting. Small models often fail in qualitatively different
ways — hallucinating labels, ignoring context, being overly verbose or terse. Capturing these
failure modes requires more than a single scalar metric.

### What the logging layer must capture per run

- Query text
- Top-k retrieved documents with scores
- Retrieved document metadata (source, chunk ID, text)
- Full prompt sent to LLM
- LLM output text
- Model name and version
- Embedding model name
- Retrieval config (k, similarity metric, index type)
- Timestamp and run ID

This makes every run reproducible and comparable. It enables post-hoc analysis of where the
pipeline fails — which is where the research value lives.

### Prior art to understand and position against

**RAGAS** — Automated RAG evaluation framework. Useful reference but depends on an LLM as the
evaluator (expensive, and circular when you're evaluating small models). Understand its metrics
but do not require it.

**TruLens** — Similar automated framework. Same limitation.

The gap both leave: they don't handle the case where your evaluator model is your subject model.
A framework that uses labeled ground truth directly, without requiring a judge LLM, is more
honest and more useful for SLM comparison work.

### The Wolf Archive as an evaluation benchmark

The pretend-play/conflict confusion documented in Xu & Hernandez (2025) is a concrete,
reproducible, expert-validated retrieval failure that should be formalized as a benchmark test
case for this pipeline. Specifically:

- The 24 "pretend-fight" episodes that S-BERT misclassified as conflict-dominant are a labeled
  negative set — cases where high cosine similarity scores do not correspond to correct retrieval
- The human expert (anthropologist) rankings provide ground truth for what the correct theme
  dominant assignment should be
- A pipeline that surfaces and correctly handles this failure mode is making a substantive claim
  about retrieval quality that goes beyond average nDCG@10

This also motivates the **asymmetric search pattern** as a first-class retrieval mode in
promptRAG, not just standard top-k retrieval:

**Standard RAG:** query → top-k most similar documents from index

**Asymmetric search (Wolf Archive pattern):** fixed anchor vectors (themes/categories) →
score every document against all anchors simultaneously → produce a distribution over
categories per document

The asymmetric pattern is particularly well-suited to the institutional text use cases where
the "query" is a fixed set of analytical categories (behavioral themes, eviction reasons, policy
categories) rather than a novel free-form question. Both patterns should be supported.

---

## 7. Module-by-Module Codebase Review

*Preserved from original research document. Confirms what needs to change and why.*

### `src/text_preprocess.py` — Keep, fix and generalize

**What works:** `clean_text()`, `remove_non_ascii()`, `remove_empty()`, `word_count_entry()`,
`get_top_n_words()` — solid utilities.

**Issues:**
- `parallelize_list()` references `np` and `Pool` but neither is imported — runtime crash
- `get_similar_words()` uses gensim word2vec; gensim is unused elsewhere and word2vec is
  superseded

**Actions:**
- Fix missing imports in `parallelize_list()` or remove it
- Drop `get_similar_words()`
- Make `clean_text()` configurable — COVID-specific regex terms must become optional/pluggable

---

### `src/create_vdb.py` — Keep structure, rebuild internals

**What works:** `IndexTextEmbeddings` class structure (init, embed, index); FAISS integration.

**Issues:**
- **Incorrect pooling**: Uses CLS token pooling, but `multi-qa-mpnet-base-dot-v1` requires mean
  pooling. Silent quality degradation. (ModernBERT and NeoBERT both use mean pooling for
  single-vector retrieval.)
- **No batching**: `create_dataset()` embeds one row at a time in a for-loop. `FIX ME` comment
  already in code. Unusable on non-trivial datasets.
- **No index persistence**: FAISS index rebuilt from scratch every run.
- **No MPS device detection**: `__init__()` (line 13-16) only checks `torch.cuda.is_available()`;
  never detects MPS. On Apple Silicon the model silently stays on CPU. Must add
  `torch.backends.mps.is_available()` check or accept a `device` parameter that defaults
  intelligently.
- **Tightly coupled to HuggingFace `datasets`**: Uses `Dataset.from_pandas()`,
  `dataset.add_faiss_index()`, and `dataset.get_nearest_examples()` — all FAISS-specific.
  Switching to ChromaDB (Decision 1) requires replacing this entire retrieval surface, not
  just swapping an index. The Plan Doc must specify how `IndexTextEmbeddings` is refactored
  to use `chromadb.Collection.add()` / `.query()` instead.

**Actions:**
- Switch to `SentenceTransformer.encode()` — handles correct pooling automatically
- Batch embedding (default `batch_size=32`, configurable)
- Add FAISS index save/load (`faiss.write_index` / `faiss.read_index`)
- Evaluate ChromaDB or LanceDB as FAISS alternatives — persistent by default, metadata
  filtering built in

---

### `src/prompt_generator.py` — Keep, fix bugs, generalize

**What works:** End-to-end query → format → template → prompt flow.

**Issues:**
- **Duplicate model loading**: `TweetPromptGenerator.__init__()` creates a new
  `IndexTextEmbeddings` instance — embedding model loaded into memory twice.
- **Loop bug in `format_retrieved_texts()`**: `texts`, `samples['code']`, `scores` variables
  are overwritten each iteration, so only the last query's raw values are returned. The
  concatenated `formatted_texts` string is correct (uses `+=`), but the returned `texts`,
  `samples['code']`, and `scores` lose all but the final iteration. Works by accident because
  `query_faiss_index()` currently always returns a single result.
- **Naming collision in `generate_prompts_for_tweets()`**: Line 48 — `tweets = tp.clean_text(tweet)`
  overwrites the outer loop's `tweets` parameter (the full input list) with a single cleaned
  string when `clean_tweets=True`. Subsequent loop iterations would fail. Currently masked
  because `example_script.py` always passes `clean_tweets=False`.
- **Hardcoded column names**: `story`, `code`, `rowid` are Poynter-specific. Not reusable.
- **Naming**: `TweetPromptGenerator`, `generate_prompts_for_tweets` — domain-specific
- **FAISS coupling in `query_faiss_index()`**: Calls `dataset.get_nearest_examples()` —
  a FAISS-specific HuggingFace `datasets` method. Must be replaced when migrating to ChromaDB.

**Actions:**
- Accept existing model instance as parameter instead of instantiating internally
- Fix loop bug in `format_retrieved_texts()`
- Column names as configurable parameters or config dict
- Rename: `RAGPipeline` or similar domain-agnostic name

---

### `src/prompt_templates/` — Rethink entirely

**Issues:**
- `promptTemplate.txt`: `[instruction]` placeholder never filled programmatically
- `promptTemplate_mistral_onthefly.txt`: `[INST]...[/INST]` format specific to Mistral v0.1/v0.2
  — incompatible with newer models
- `{tweet}` placeholder: domain-specific

**Actions:**
- Use tokenizer's `apply_chat_template()` for model-specific formatting
- Jinja2 for custom templates where needed
- System prompt and instruction configurable at runtime, not baked into template files
- Generic `{input_text}` as the query placeholder

---

### `src/transformer_playground.py` — Drop entirely

- Educational multi-head attention implementation, not part of the pipeline
- Inline script execution at module level (lines 283+) — importing this module runs code
- References `plot_multihead_attention()` never defined in the file
- Move to a standalone notebook or separate repo if worth keeping

---

### `example_script.py` — Replace

- Hardcoded paths, debug output, unused imports (`gensim`, `json`, `operator`)
- Currently the only entrypoint — no CLI, no config, no argument parsing

**Actions:**
- Replace with config-driven `main.py` reading from YAML/TOML
- Add CLI entrypoint (`click` preferred over `argparse` for usability)

---

## 8. Vector Store Decision

**Current:** FAISS via HuggingFace `datasets`. Functional but no persistence.

**Options:**

| Store | Persistence | Metadata filtering | Local | Notes |
|---|---|---|---|---|
| FAISS | Manual save/load | No | Yes | Fastest; needs wrapper code for persistence |
| ChromaDB | Built-in | Yes | Yes | Simplest local option; good Python API |
| LanceDB | Built-in | Yes | Yes | Columnar storage; good for large datasets |
| Qdrant | Built-in | Yes | Local or cloud | More infrastructure overhead |

**Recommendation:** Evaluate ChromaDB first — lowest friction for a research team setup.
Keep FAISS as a fallback for pure speed benchmarking. The evaluation layer needs metadata
filtering (run ID, dataset name, model name) which FAISS doesn't support natively.

---

## 9. Dependencies and Environment

### Current state (do not carry over)

- Python 3.8.13 — EOL since October 2024
- 160+ pinned packages including system-level libs — not portable
- PyTorch 1.12.1 (2022), transformers 4.24.0 (Nov 2022), sentence-transformers 2.2.2

### Target

- **Python 3.11+**
- `pyproject.toml` with direct dependencies only (not transitive)
- `uv` for environment management (fast, reproducible)

### Minimum direct dependencies

```
torch>=2.5
transformers>=4.47
sentence-transformers>=3.0
chromadb               # vector store (Decision 1 — replaces faiss-cpu)
numpy>=1.26
polars                 # DataFrame library (Decision 4 — replaces pandas)
click                  # CLI
pyyaml                 # config
jinja2                 # prompt templates
```

**Note on `datasets` (HuggingFace):** The current code uses `from datasets import Dataset` for
both DataFrame-to-Dataset conversion and FAISS integration (`add_faiss_index`,
`get_nearest_examples`). Switching to ChromaDB removes the FAISS dependency, and switching to
Polars removes the pandas/Dataset pipeline. The `datasets` library is no longer a direct
dependency unless needed for data loading from HuggingFace Hub. Drop from minimum deps; add
to optional if Hub loading is used.

**Note on `pandas`:** Listed as `pandas>=2.0` in the original dep list, but Decision 4 selects
Polars as the DataFrame library. Pandas may still be needed transiently (e.g., CSV loading
before Polars migration, or libraries that return pandas DataFrames), but it should not be
the primary analysis tool. Keep as optional, not minimum.

### Optional / evaluation dependencies

```
ragas                  # reference, not required
mlflow                 # or wandb for experiment tracking
pytest                 # testing
connectorx             # or adbc-driver-sqlite — required for Polars read_database
datasets               # HuggingFace Hub data loading (optional)
pandas                 # transitional / compatibility (optional)
```

---

## 10. Files: Keep / Drop / Build

| File | Decision | Notes |
|---|---|---|
| `src/text_preprocess.py` | **Keep** | Fix imports, generalize `clean_text()` |
| `src/create_vdb.py` | **Keep** | Fix pooling, add batching, add persistence |
| `src/prompt_generator.py` | **Keep** | Fix loop bug, generalize naming/columns |
| `src/transformer_playground.py` | **Drop** | Not pipeline code |
| `src/prompt_templates/*.txt` | **Rethink** | Replace with chat template + Jinja2 approach |
| `example_script.py` | **Replace** | Build config-driven CLI entrypoint |
| `example_notebook.ipynb` | **Keep** | Update alongside code |
| `environment.yml` | **Drop** | Rebuild as `pyproject.toml` |
| `src/__init__.py` | **Keep** | Update exports to match refactored module names |
| — | **Build** | `src/llm_backend.py` — pluggable LLM interface |
| — | **Build** | `src/evaluation.py` — retrieval + generation metrics |
| — | **Build** | `src/chunker.py` — configurable document chunking |
| — | **Build** | `configs/` — YAML config templates per dataset |
| — | **Build** | `experiments/` — logged run outputs |

---

## 11. Architectural Decisions

All five blocking decisions are resolved. These are locked in and inform the implementation plan.

### Decision 1: Vector Store — ChromaDB ✓

FAISS rejected. ChromaDB selected for the following reasons:
- Persistence is built in — no wrapper code required
- Native metadata filtering — filter by `run_id`, `embedding_model`, `dataset` without
  extra machinery. Required by the evaluation layer.
- Collections map cleanly to dataset + embedding model combinations
- At the document scale of this project (hundreds to low thousands), FAISS's speed advantage
  is irrelevant

**Note:** Pin ChromaDB version explicitly. The library has a history of breaking API changes
between releases.

---

### Decision 2: Embedding Model as Experiment Variable ✓

The embedding model is a **first-class experiment variable**, not a fixed config value.

Rationale: answering whether retrieval quality or generation quality is the bottleneck requires
holding one constant and varying the other. That means running the same dataset through
ModernBERT and NeoBERT and comparing downstream results cleanly. Building this in now is
two hours of design work; bolting it on later means touching ChromaDB schema, the logging
layer, and the config system simultaneously.

**ChromaDB collection naming convention:**
```
{dataset}_{embedding_model_shortname}

wolf_archive_modernbert_base
wolf_archive_neobert
eviction_modernbert_base
eviction_neobert
covid_modernbert_base
covid_neobert
```

Each collection is a specific dataset + embedding model combination. The SQLite experiments
table ties runs to collections. Cross-run queries can isolate embedding model as a variable.

**Minimum run config (YAML):**
```yaml
embedding_model: answerdotai/ModernBERT-base
llm_model: microsoft/Phi-3-mini-4k-instruct
dataset: wolf_archive
k: 5
```

Both model fields are logged to SQLite on every run so any result is fully reproducible.

---

### Decision 3: Chunking Strategy — Deferred ✓

Start with configurable fixed-size chunking. The eviction documents (long-form legal text)
and Wolf Archive observations (short, ~250 words) will need different chunk size profiles,
but this does not need to be solved in phase 1. Revisit when both datasets are active.

**Why fixed-size first:** The three common strategies — fixed-size (with overlap), sentence-based
(spaCy/NLTK sentence segmentation), and recursive character splitting (LangChain-style) — all
have trade-offs. Fixed-size is the simplest to implement and reason about, and it exposes
chunking as a controlled variable for evaluation: if retrieval quality is bad, you know it's
not because of a complex chunking heuristic. Sentence-based splitting is the likely phase 2
upgrade, especially for the eviction records where legal sentences carry self-contained meaning.
Semantic chunking (embedding-based boundary detection) is expensive and hard to evaluate
independently — defer until baseline results justify the complexity.

**Dataset-specific consideration:** Wolf Archive observations (~250 words) are often smaller
than a single chunk at typical sizes (512 tokens). These may not need chunking at all — they
can be embedded as whole documents. The chunker must handle the "document smaller than chunk
size" case cleanly (pass through, do not pad or split).

---

### Decision 4: Evaluation Logging — SQLite + Polars ✓

**Format:** SQLite — single local file, no infrastructure, queryable with SQL.

Rationale: once you are comparing five models across three datasets, you need to query that
data, not manually load and join files. `SELECT * FROM runs WHERE embedding_model =
'modernbert_base' AND dataset = 'wolf_archive'` is exactly the query the research workflow
requires.

**DataFrame library:** Polars over pandas. Polars reads from SQLite directly via
`read_database` with a connection string. Lazy evaluation pushes filters down before loading
into memory — the right default for a growing experiments table.

**Connector dependency:** `connectorx` or `adbc-driver-sqlite` required for Polars
`read_database`. Pin explicitly.

**Minimum SQLite run record schema:**
```
run_id          TEXT PRIMARY KEY
timestamp       TEXT
dataset         TEXT
embedding_model TEXT
llm_model       TEXT
k               INTEGER
query           TEXT
retrieved_docs  TEXT    -- JSON serialized list of chunk text
retrieved_ids   TEXT    -- JSON serialized list of chunk IDs
scores          TEXT    -- JSON serialized list of similarity scores
prompt          TEXT    -- full prompt sent to LLM
llm_output      TEXT
notes           TEXT    -- optional free-text annotation
```

---

### Decision 5: LLM Backend — HuggingFace First, MPS Backend ✓

**Primary backend:** HuggingFace `transformers` pipeline.

Rationale: SLM evaluation requires access to model internals (logits, token probabilities)
that Ollama abstracts away. HuggingFace also covers every model without waiting for Ollama
packaging, and integrates naturally with sentence-transformers in the same ecosystem.

**Hardware:** MacBook Pro M4 Max with unified memory (48GB or 64GB). Use `device_map="mps"`
to route inference through Apple's Metal Performance Shaders backend.

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device_map="mps"
)
```

**MPS fallback:** Enable for operations not yet implemented in the Metal backend:
```python
import torch
torch.backends.mps.enable_fallback_for_missing_ops = True
```

**Ollama:** Not in phase 1. Design the LLM backend interface cleanly enough that Ollama
can be added later as a convenience backend for quick runs and collaborator use. Adding it
should be an afternoon of work, not a refactor.

**Priority SLMs to evaluate** (all run well on M4 Max unified memory):
- `microsoft/Phi-3-mini-4k-instruct` (3.8B)
- `Qwen/Qwen2.5-7B-Instruct` (7B)
- `google/gemma-2-2b-it` (2B)
- `mistralai/Mistral-7B-Instruct-v0.3` (7B)

**Note (April 2026):** This list was compiled in early 2025. By implementation time, newer
versions likely exist (Phi-4, Qwen3, Gemma 3, Mistral successors). Refresh the list against
current HuggingFace model cards before the first evaluation run. The model families remain
correct as targets; the specific version tags should be updated.

---

## 12. Relevant Prior Art

**Xu, J. & Hernandez, J. M. (2025)** — "Reading children's moral dramas in anthropological
fieldnotes: A human–AI hybrid approach." *Cambridge Forum on AI: Culture and Society*, 1, e6.
doi:10.1017/cfc.2025.10008. The origin implementation of this pipeline's semantic search
approach. Documents the `multi-qa-mpnet-base-dot-v1` + FAISS + asymmetric search design on
the Wolf Archive CO corpus. Provides the pretend-play/conflict failure mode as a concrete
evaluation benchmark. Code: github.com/jmhernan/TaiwanChildhoodStudy

**ModernBERT** (Warner et al., 2024) — `arXiv:2412.13663`
State-of-the-art encoder-only model. Recommended default embedding backbone. See Section 4.

**NeoBERT** (Le Breton et al., 2025) — `arXiv:2502.19587`
Concurrent modernization from Mila. Strong MTEB results, faster inference, fully open release.
Direct competitor/comparison point for ModernBERT in the embedding evaluation. See Section 4.

**RAGAS** — Automated RAG evaluation framework. Useful reference for metric design; dependency
on a judge LLM limits direct use for SLM evaluation. Understand its gaps.

**BEIR** (Thakur et al., 2021) — Standard heterogeneous retrieval benchmark used by both
ModernBERT and NeoBERT papers. nDCG@10 is the standard retrieval metric here.

**MTEB** — Massive Text Embedding Benchmark (56 datasets, 7 task types). Used by NeoBERT as
primary evaluation suite. More comprehensive than BEIR for embedding model comparison.

**sentence-transformers** (Reimers & Gurevych, 2019) — The fine-tuning framework used by
ModernBERT for retrieval. The `SentenceTransformer.encode()` method handles pooling correctly
and should replace the current manual CLS pooling implementation.

**ColBERT** (Khattab & Zaharia, 2020) — Multi-vector retrieval using all token representations
rather than a single CLS vector. ModernBERT shows exceptional ColBERT performance (80+ nDCG
on MLDR). Worth evaluating as a retrieval strategy alongside single-vector DPR, especially for
long institutional documents.

**Cross-encoder reranking** — A standard RAG improvement step where a cross-encoder model
(e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores the top-k retrieved documents by
attending jointly to the query and each candidate. Particularly relevant for the
precision-oriented institutional text use case where the initial retrieval may surface
phraseologically similar but semantically wrong documents (cf. the Wolf Archive pretend-play
failure). Not in phase 1, but the retrieval interface should be designed so a reranking step
can be inserted between retrieval and generation without refactoring. The evaluation layer
should be able to compare runs with and without reranking to measure its marginal contribution.