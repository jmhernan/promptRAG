# promptRAG — Plan Document

*Phase 1: Full pipeline — Wolf Archive (ingestion + generation) + COVID/Poynter (retrieval eval metrics).*
*Reference: `docs/research.md` (April 2026)*

---

## Scope

### What is in Phase 1

- Proper package structure (`src/promptrag/`)
- Embedding module (SentenceTransformer, ModernBERT-base default, MPS)
- ChromaDB vector store module (persistence, metadata filtering)
- Fixed-size chunker with whole-document pass-through
- HuggingFace `transformers` LLM backend (MPS, `apply_chat_template`)
- Jinja2-based prompt engine
- Evaluation module (retrieval metrics + generation metrics + SQLite logging)
- Pipeline orchestrator tying all modules together
- Click CLI entrypoint
- YAML configs for Wolf Archive and COVID/Poynter
- COVID/Poynter retrieval evaluation (labeled data available — `code` column)
- Wolf Archive ingestion + generation testing (unlabeled — retrieval eval deferred)
- Moderate pytest coverage for key functions
- `pyproject.toml` with `uv` for environment management

### What is NOT in Phase 1

- Ollama backend (deferred — interface designed for it, not implemented)
- vLLM backend
- OpenAI / Anthropic API backends
- Eviction records dataset wiring
- Wolf Archive gold set labeling (Phase 2 — see labeling note below)
- Cross-encoder reranking
- ColBERT multi-vector retrieval
- Sentence-based or semantic chunking
- MLflow / WandB experiment tracking
- CI pipeline
- Fine-tuning of any models

---

## Pre-Implementation Steps

### Step 0.1: Archive existing code

```bash
git checkout -b archive/legacy
git checkout main
```

The existing `src/` files are preserved in the `archive/legacy` branch for reference.
New modules are built from scratch in `src/promptrag/`.

### Step 0.2: Dataset inventory

**Wolf Archive** — already in `data/4ChildObservation_MasterFile.csv`:
- 1,678 observations (not 1,668 as cited in the paper — minor discrepancy, likely includes
  additional observations not used in the published analysis)
- Text column: `text` (word count: min 1, max 1,434, median 306, mean 336)
- ID column: `Index`
- Other columns: `Page`, `TargetPerson`, `Location`, `Date`, `Time`,
  `Duration (min)`, `present1`–`present12`
- **No theme/behavior labels.** The pretend-play/conflict classifications from Xu & Hernandez
  (2025) are not present in this file. Labels would need to be created from scratch.

**COVID/Poynter** — already in `data/poynter_coded_breon_tab.csv`:
- 316 rows, tab-separated
- Text column: `story` (raw) / `story_copy` (clean)
- ID column: `rowid`
- Label column: `code` — misinformation category labels (human-coded)
- This dataset has labels and can support retrieval evaluation metrics immediately.

**Labeling strategy for Wolf Archive (Phase 2):**
The 24 pretend-play episodes documented in Xu & Hernandez (2025) need to be identified in
this dataset by `Index` and manually labeled. This can be done with agent assistance — the
observations contain identifiable content ("Kill that Strange Bird" etc.) that can be
located via text search. This is scoped out of Phase 1 but is the first task in Phase 2.

---

## Implementation Plan

### Step 1: Package scaffolding

**Goal:** Set up project structure, `pyproject.toml`, and empty module files.

**Creates:**
```
pyproject.toml
src/promptrag/__init__.py
src/promptrag/embeddings.py
src/promptrag/vector_store.py
src/promptrag/chunker.py
src/promptrag/llm_backend.py
src/promptrag/prompt_engine.py
src/promptrag/evaluation.py
src/promptrag/pipeline.py
src/promptrag/cli.py
configs/wolf_archive.yaml
configs/covid_poynter.yaml
experiments/             # directory only — SQLite db created at runtime
tests/__init__.py
tests/test_embeddings.py
tests/test_vector_store.py
tests/test_chunker.py
tests/test_evaluation.py
```

**Removes (from working tree, preserved in `archive/legacy`):**
```
src/__init__.py
src/create_vdb.py
src/prompt_generator.py
src/text_preprocess.py
src/transformer_playground.py
src/prompt_templates/promptTemplate.txt
src/prompt_templates/promptTemplate_mistral_onthefly.txt
example_script.py
environment.yml
```

**Updates:** `.gitignore` — add:
```
# Experiments (generated at runtime)
experiments/
```

**`pyproject.toml`:**
```toml
[project]
name = "promptrag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.5",
    "transformers>=4.47",
    "sentence-transformers>=3.0",
    "chromadb>=0.5",
    "numpy>=1.26",
    "polars>=1.0",
    "click>=8.0",
    "pyyaml>=6.0",
    "jinja2>=3.1",
    "connectorx>=0.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pandas>=2.0",
]

[project.scripts]
promptrag = "promptrag.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/promptrag"]
```

**`src/promptrag/__init__.py`:**
```python
"""promptRAG — Research-team-grade RAG pipeline for institutional text."""
```

**Test point:** `uv pip install -e ".[dev]"` succeeds. `python -c "import promptrag"` succeeds.

---

### Step 2: YAML config

**Goal:** Define the run configuration schema and create the Wolf Archive config.

**Creates:** `configs/wolf_archive.yaml`

**Creates:** `configs/wolf_archive.yaml`

```yaml
dataset:
  name: wolf_archive
  path: data/4ChildObservation_MasterFile.csv
  text_column: text
  id_column: Index
  # No gold set in Phase 1 — labels do not exist yet.
  # gold_set_path: data/wolf_archive_gold.csv  # Phase 2

embedding:
  model: answerdotai/ModernBERT-base
  device: mps
  batch_size: 32

chunking:
  strategy: fixed
  chunk_size: 512                    # word count (not tokens) — Wolf Archive median is 306, most pass through
  chunk_overlap: 50

vector_store:
  provider: chromadb
  persist_directory: experiments/chromadb
  collection_name: wolf_archive_modernbert_base  # per naming convention

retrieval:
  k: 5
  similarity_metric: cosine

llm:
  model: microsoft/Phi-3-mini-4k-instruct
  device_map: mps
  max_new_tokens: 256

evaluation:
  db_path: experiments/runs.db
```

**Creates:** `configs/covid_poynter.yaml`

```yaml
dataset:
  name: covid
  path: data/poynter_coded_breon_tab.csv
  text_column: story_copy
  id_column: rowid
  label_column: code                 # misinformation category — used for retrieval eval
  separator: "\t"                    # tab-separated

embedding:
  model: answerdotai/ModernBERT-base
  device: mps
  batch_size: 32

chunking:
  strategy: fixed
  chunk_size: 512                    # word count (not tokens)
  chunk_overlap: 50

vector_store:
  provider: chromadb
  persist_directory: experiments/chromadb
  collection_name: covid_modernbert_base

retrieval:
  k: 5
  similarity_metric: cosine

llm:
  model: microsoft/Phi-3-mini-4k-instruct
  device_map: mps
  max_new_tokens: 256

evaluation:
  db_path: experiments/runs.db
```

**Config loading code** (in `src/promptrag/pipeline.py`, used by CLI):
```python
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)
```

**Test point:** Both configs load without error. All required keys present.

---

### Step 3: Embedding module

**Goal:** Wrap SentenceTransformer with MPS support, batch encoding, and correct pooling.

**Creates:** `src/promptrag/embeddings.py`

```python
import torch
torch.backends.mps.enable_fallback_for_missing_ops = True

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wraps SentenceTransformer with device detection and batch encoding."""

    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", device: str = "mps"):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode texts using SentenceTransformer.encode() — handles pooling correctly."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
```

**Key design decisions:**
- `SentenceTransformer.encode()` handles mean pooling automatically — fixes the CLS pooling
  bug from `create_vdb.py` line 24
- MPS detection with fallback — fixes the CUDA-only check from `create_vdb.py` line 14-16
- `encode()` returns `list[list[float]]` — ChromaDB's expected input format
- No manual tokenization or model forward pass — that was the source of the pooling bug

**Test point (`tests/test_embeddings.py`):**
```python
def test_embedding_dimension():
    model = EmbeddingModel(device="cpu")
    result = model.encode(["test sentence"])
    assert len(result) == 1
    assert len(result[0]) == model.dimension

def test_batch_encoding():
    model = EmbeddingModel(device="cpu")
    texts = [f"sentence {i}" for i in range(10)]
    result = model.encode(texts, batch_size=4)
    assert len(result) == 10

def test_device_fallback():
    model = EmbeddingModel(device="cuda")  # not available in test env
    assert model.device == "cpu"
```

---

### Step 4: Chunker

**Goal:** Fixed-size text chunking with pass-through for short documents.

**Creates:** `src/promptrag/chunker.py`

```python
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict


def chunk_documents(
    texts: list[str],
    doc_ids: list[str],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    metadata: list[dict] | None = None,
) -> list[Chunk]:
    """Split documents into fixed-size chunks by word count.

    Documents shorter than chunk_size are passed through as a single chunk.
    """
    chunks = []
    for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
        doc_meta = metadata[i] if metadata else {}
        words = text.split()

        if len(words) <= chunk_size:
            chunks.append(Chunk(
                text=text,
                doc_id=doc_id,
                chunk_index=0,
                metadata=doc_meta,
            ))
            continue

        start = 0
        chunk_idx = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=chunk_idx,
                metadata=doc_meta,
            ))
            start += chunk_size - chunk_overlap
            chunk_idx += 1

    return chunks
```

**Key design decisions:**
- Word-count based, not token-count — simpler, model-agnostic. Token-count chunking can
  be added as a strategy option later.
- Wolf Archive docs (~250 words) will pass through as whole documents at `chunk_size=512`.
  This is intentional — they are short enough to embed as-is.
- `Chunk` dataclass carries `doc_id` and `chunk_index` for ChromaDB metadata.

**Test point (`tests/test_chunker.py`):**
```python
def test_short_document_passthrough():
    chunks = chunk_documents(["short text"], ["doc1"], chunk_size=512)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].chunk_index == 0

def test_long_document_split():
    long_text = " ".join(["word"] * 1000)
    chunks = chunk_documents([long_text], ["doc1"], chunk_size=512, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c.doc_id == "doc1" for c in chunks)

def test_overlap():
    text = " ".join(["word"] * 100)
    chunks = chunk_documents([text], ["doc1"], chunk_size=60, chunk_overlap=10)
    # With 100 words, chunk_size=60, overlap=10: chunks at [0:60], [50:100]
    assert len(chunks) == 2
```

---

### Step 5: Vector store module

**Goal:** ChromaDB wrapper — create/load collections, add documents, query.

**Creates:** `src/promptrag/vector_store.py`

```python
import chromadb
from promptrag.embeddings import EmbeddingModel
from promptrag.chunker import Chunk


class VectorStore:
    """ChromaDB-backed vector store with embedding model integration."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_directory: str = "experiments/chromadb",
    ):
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=persist_directory)

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a named collection. Name follows convention:
        {dataset}_{embedding_model_shortname}
        """
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, collection: chromadb.Collection, chunks: list[Chunk]) -> None:
        """Add chunked documents to a collection with metadata."""
        ids = [f"{c.doc_id}_chunk{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"doc_id": c.doc_id, "chunk_index": c.chunk_index, **c.metadata}
            for c in chunks
        ]
        embeddings = self.embedding_model.encode(documents)

        # ChromaDB has a batch limit — add in batches of 5000
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def query(
        self,
        collection: chromadb.Collection,
        query_texts: list[str],
        k: int = 5,
    ) -> dict:
        """Query collection. Returns ChromaDB results dict with
        ids, documents, distances, metadatas.
        """
        query_embeddings = self.embedding_model.encode(query_texts)
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
```

**Key design decisions:**
- `PersistentClient` — data survives across runs (fixes no-persistence bug from `create_vdb.py`)
- Embedding happens inside `add_chunks()` — single responsibility, no separate embed-then-index step
- Collection naming follows research.md convention: `wolf_archive_modernbert_base`
- `query()` returns raw ChromaDB results — the pipeline layer handles formatting
- Replaces the entire `IndexTextEmbeddings` class and `TweetPromptGenerator.query_faiss_index()`

**Test point (`tests/test_vector_store.py`):**
```python
import tempfile

def test_add_and_query(tmp_path):
    model = EmbeddingModel(device="cpu")
    store = VectorStore(model, persist_directory=str(tmp_path / "chroma"))
    collection = store.get_or_create_collection("test_collection")

    chunks = [
        Chunk(text="the cat sat on the mat", doc_id="d1", chunk_index=0, metadata={}),
        Chunk(text="dogs love to play fetch", doc_id="d2", chunk_index=0, metadata={}),
    ]
    store.add_chunks(collection, chunks)

    results = store.query(collection, ["feline sitting"], k=1)
    assert results["ids"][0][0] == "d1_chunk0"

def test_collection_persistence(tmp_path):
    model = EmbeddingModel(device="cpu")
    persist_dir = str(tmp_path / "chroma")

    store1 = VectorStore(model, persist_directory=persist_dir)
    coll1 = store1.get_or_create_collection("persist_test")
    chunks = [Chunk(text="persistence test", doc_id="d1", chunk_index=0, metadata={})]
    store1.add_chunks(coll1, chunks)

    # New client instance, same directory — data should persist
    store2 = VectorStore(model, persist_directory=persist_dir)
    coll2 = store2.get_or_create_collection("persist_test")
    assert coll2.count() == 1
```

---

### Step 6: LLM backend

**Goal:** HuggingFace `transformers` pipeline wrapper with MPS support and `apply_chat_template`.

**Creates:** `src/promptrag/llm_backend.py`

```python
import torch
torch.backends.mps.enable_fallback_for_missing_ops = True

from transformers import AutoTokenizer, pipeline as hf_pipeline


class HuggingFaceLLM:
    """HuggingFace transformers pipeline for text generation."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device_map: str = "mps",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device_map=self._resolve_device_map(device_map),
            torch_dtype=torch.float16,
        )

    @staticmethod
    def _resolve_device_map(requested: str) -> str:
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cuda" and torch.cuda.is_available():
            return "auto"
        return "cpu"

    def generate(self, prompt: str) -> str:
        """Generate text from a formatted prompt string."""
        result = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return result[0]["generated_text"]

    def generate_from_messages(self, messages: list[dict]) -> str:
        """Generate from a chat messages list. Applies the model's chat template."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt)
```

**Key design decisions:**
- `apply_chat_template()` instead of hand-rolled `[INST]...[/INST]` templates — fixes the
  Mistral-specific template problem from `promptTemplate_mistral_onthefly.txt`
- `generate_from_messages()` takes a standard `messages` list — model-agnostic interface
- `do_sample=False` for reproducible evaluation runs (deterministic greedy decoding)
- `torch.float16` — reduces memory footprint on M4 Max. Can be made configurable if needed.
- No Ollama in phase 1. The interface (`generate(prompt)` / `generate_from_messages(messages)`)
  is clean enough that an `OllamaLLM` class with the same two methods can be added later.

**Test point:** Manual — load a small model (`google/gemma-2-2b-it`), generate from a simple
prompt, verify output is non-empty and deterministic across two calls. Full model loading is
too heavy for automated unit tests.

---

### Step 7: Prompt engine

**Goal:** Jinja2-based prompt construction with configurable system prompts and retrieval context.

**Creates:** `src/promptrag/prompt_engine.py`

```python
from jinja2 import Environment, BaseLoader


DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question based only on the "
    "provided context. If the context does not contain enough information "
    "to answer, say so."
)

DEFAULT_USER_TEMPLATE = (
    "Context:\n"
    "{% for doc in retrieved_docs %}"
    "- {{ doc }}\n"
    "{% endfor %}\n\n"
    "Question: {{ query }}"
)


class PromptEngine:
    """Build chat messages from retrieved context using Jinja2 templates."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str = DEFAULT_USER_TEMPLATE,
    ):
        self.system_prompt = system_prompt
        self.env = Environment(loader=BaseLoader())
        self.user_template = self.env.from_string(user_template)

    def build_messages(
        self,
        query: str,
        retrieved_docs: list[str],
    ) -> list[dict]:
        """Build a chat messages list for the LLM backend."""
        user_content = self.user_template.render(
            query=query,
            retrieved_docs=retrieved_docs,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
```

**Key design decisions:**
- Returns `list[dict]` messages — passed to `HuggingFaceLLM.generate_from_messages()` which
  applies the correct chat template per model. This separates content from formatting.
- System prompt is a parameter, not baked into a file — fixes `promptTemplate.txt` where
  `[instruction]` was never filled
- Jinja2 renders the user message — supports conditional logic, loops, custom formatting
- `{tweet}` → `{{ query }}` — generic, not domain-specific
- Templates can be overridden per config or per dataset

**Test point:** Unit test — build messages, verify structure:
```python
def test_build_messages():
    engine = PromptEngine()
    messages = engine.build_messages("what happened?", ["doc1 text", "doc2 text"])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "doc1 text" in messages[1]["content"]
    assert "what happened?" in messages[1]["content"]
```

---

### Step 8: Evaluation module

**Goal:** Retrieval metrics, generation logging, SQLite persistence.

**Creates:** `src/promptrag/evaluation.py`

```python
import json
import sqlite3
import uuid
from datetime import datetime, timezone


# --- Retrieval metrics ---

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@k: fraction of top-k retrieved docs that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k: fraction of relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return len(set(top_k) & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    import math
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(i + 2) for i, doc_id in enumerate(top_k)
        if doc_id in relevant_ids
    )
    ideal_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# --- SQLite run logging ---

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    llm_model       TEXT NOT NULL,
    k               INTEGER NOT NULL,
    query           TEXT NOT NULL,
    retrieved_docs  TEXT NOT NULL,
    retrieved_ids   TEXT NOT NULL,
    scores          TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    llm_output      TEXT NOT NULL,
    notes           TEXT
)
"""


class RunLogger:
    """Log experiment runs to SQLite."""

    def __init__(self, db_path: str = "experiments/runs.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(CREATE_TABLE_SQL)
        self.conn.commit()

    def log_run(
        self,
        dataset: str,
        embedding_model: str,
        llm_model: str,
        k: int,
        query: str,
        retrieved_docs: list[str],
        retrieved_ids: list[str],
        scores: list[float],
        prompt: str,
        llm_output: str,
        notes: str = "",
    ) -> str:
        run_id = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO runs
            (run_id, timestamp, dataset, embedding_model, llm_model, k,
             query, retrieved_docs, retrieved_ids, scores, prompt, llm_output, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                dataset,
                embedding_model,
                llm_model,
                k,
                query,
                json.dumps(retrieved_docs),
                json.dumps(retrieved_ids),
                json.dumps(scores),
                prompt,
                llm_output,
                notes,
            ),
        )
        self.conn.commit()
        return run_id

    def close(self):
        self.conn.close()
```

**Key design decisions:**
- SQLite schema matches research.md Decision 4 exactly — all 13 columns present
- Parameterized queries — no SQL injection risk
- `run_id` is a UUID, not auto-increment — safe for concurrent use, portable
- Retrieval metrics are pure functions — testable without a database, composable
- Binary relevance for nDCG — matches the Wolf Archive gold set (relevant vs. not relevant)
- `RunLogger` uses context-free `sqlite3` — no ORM, no infrastructure

**Test point (`tests/test_evaluation.py`):**
```python
def test_precision_at_k():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d1", "d3", "d5"}
    assert precision_at_k(retrieved, relevant, 5) == 0.6
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2/3)

def test_recall_at_k():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d3", "d5"}
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2/3)

def test_mrr():
    assert mrr(["d2", "d1", "d3"], {"d1"}) == 0.5
    assert mrr(["d1", "d2", "d3"], {"d1"}) == 1.0
    assert mrr(["d2", "d3"], {"d1"}) == 0.0

def test_ndcg_at_k():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}
    score = ndcg_at_k(retrieved, relevant, 3)
    assert 0 < score <= 1.0

def test_run_logger(tmp_path):
    logger = RunLogger(db_path=str(tmp_path / "test.db"))
    run_id = logger.log_run(
        dataset="wolf_archive",
        embedding_model="modernbert_base",
        llm_model="phi3",
        k=5,
        query="test query",
        retrieved_docs=["doc1"],
        retrieved_ids=["id1"],
        scores=[0.95],
        prompt="test prompt",
        llm_output="test output",
    )
    assert run_id is not None

    # Verify retrievable
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row is not None
    conn.close()
    logger.close()
```

---

### Step 9: Pipeline orchestrator

**Goal:** Tie all modules together into a single run flow: load data → chunk → embed →
store → query → prompt → generate → log.

**Creates:** `src/promptrag/pipeline.py`

```python
import yaml
import polars as pl
from pathlib import Path

from promptrag.embeddings import EmbeddingModel
from promptrag.vector_store import VectorStore
from promptrag.chunker import chunk_documents
from promptrag.llm_backend import HuggingFaceLLM
from promptrag.prompt_engine import PromptEngine
from promptrag.evaluation import RunLogger


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class RAGPipeline:
    """End-to-end RAG pipeline: ingest, retrieve, generate, log."""

    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = EmbeddingModel(
            model_name=config["embedding"]["model"],
            device=config["embedding"]["device"],
        )
        self.vector_store = VectorStore(
            embedding_model=self.embedding_model,
            persist_directory=config["vector_store"]["persist_directory"],
        )
        self.llm = HuggingFaceLLM(
            model_name=config["llm"]["model"],
            device_map=config["llm"]["device_map"],
            max_new_tokens=config["llm"]["max_new_tokens"],
        )
        self.prompt_engine = PromptEngine()
        self.logger = RunLogger(db_path=config["evaluation"]["db_path"])
        self.collection = self.vector_store.get_or_create_collection(
            config["vector_store"]["collection_name"]
        )

    def ingest(self) -> int:
        """Load dataset, chunk, embed, and store. Returns chunk count."""
        ds_config = self.config["dataset"]
        df = pl.read_csv(
            ds_config["path"],
            separator=ds_config.get("separator", ","),
        )

        # Drop rows with null text — COVID story_copy may have nulls
        df = df.filter(pl.col(ds_config["text_column"]).is_not_null())

        texts = df[ds_config["text_column"]].to_list()
        ids = [str(x) for x in df[ds_config["id_column"]].to_list()]

        chunks = chunk_documents(
            texts=texts,
            doc_ids=ids,
            chunk_size=self.config["chunking"]["chunk_size"],
            chunk_overlap=self.config["chunking"]["chunk_overlap"],
        )

        self.vector_store.add_chunks(self.collection, chunks)
        return len(chunks)

    def query(self, query_text: str, k: int | None = None) -> dict:
        """Run a single query through the full pipeline. Returns run record."""
        k = k or self.config["retrieval"]["k"]

        # Retrieve
        results = self.vector_store.query(self.collection, [query_text], k=k)
        retrieved_docs = results["documents"][0]
        retrieved_ids = results["ids"][0]
        distances = results["distances"][0]
        # ChromaDB cosine distance -> similarity score
        scores = [1.0 - d for d in distances]

        # Build prompt and generate
        messages = self.prompt_engine.build_messages(query_text, retrieved_docs)
        llm_output = self.llm.generate_from_messages(messages)

        # Reconstruct prompt string for logging
        prompt_str = self.llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Log
        run_id = self.logger.log_run(
            dataset=self.config["dataset"]["name"],
            embedding_model=self.config["embedding"]["model"],
            llm_model=self.config["llm"]["model"],
            k=k,
            query=query_text,
            retrieved_docs=retrieved_docs,
            retrieved_ids=retrieved_ids,
            scores=scores,
            prompt=prompt_str,
            llm_output=llm_output,
        )

        return {
            "run_id": run_id,
            "query": query_text,
            "retrieved_docs": retrieved_docs,
            "retrieved_ids": retrieved_ids,
            "scores": scores,
            "llm_output": llm_output,
        }

    def close(self):
        self.logger.close()
```

**Key design decisions:**
- `ingest()` and `query()` are separate — ingest once, query many times
- Polars `read_csv` replaces pandas — per Decision 4
- ChromaDB cosine distance is `1 - similarity` — conversion happens here, scores in SQLite
  are similarity values (higher = more similar)
- All run data logged automatically — every query is an experiment record
- Column names come from config, not hardcoded — fixes the `story`/`code`/`rowid` coupling

**Test point:** Integration test with Wolf Archive data — ingest, query, verify SQLite record
exists with correct schema.

---

### Step 10: CLI entrypoint

**Goal:** Click-based CLI for `ingest` and `query` commands.

**Creates:** `src/promptrag/cli.py`

```python
import click
from promptrag.pipeline import RAGPipeline, load_config


@click.group()
def main():
    """promptRAG — Research-team-grade RAG pipeline."""
    pass


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
def ingest(config):
    """Ingest dataset into vector store."""
    cfg = load_config(config)
    pipeline = RAGPipeline(cfg)
    count = pipeline.ingest()
    click.echo(f"Ingested {count} chunks into {cfg['vector_store']['collection_name']}")
    pipeline.close()


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
@click.option("--k", default=None, type=int, help="Number of documents to retrieve")
@click.argument("query_text")
def query(config, k, query_text):
    """Run a query through the pipeline."""
    cfg = load_config(config)
    pipeline = RAGPipeline(cfg)
    result = pipeline.query(query_text, k=k)
    click.echo(f"\nRun ID: {result['run_id']}")
    click.echo(f"\nRetrieved {len(result['retrieved_docs'])} documents:")
    for doc_id, score, text in zip(result["retrieved_ids"], result["scores"], result["retrieved_docs"]):
        click.echo(f"  [{doc_id}] (score: {score:.4f}) {text[:100]}...")
    click.echo(f"\nLLM Output:\n{result['llm_output']}")
    pipeline.close()
```

**Test point:** `promptrag --help` shows `ingest` and `query` commands.
`promptrag ingest --config configs/wolf_archive.yaml` ingests the dataset.
`promptrag query --config configs/wolf_archive.yaml "what types of play are observed?"` returns results.

---

### Step 11: End-to-end evaluation run (COVID/Poynter)

**Goal:** Run the full pipeline on COVID/Poynter and exercise the retrieval evaluation
metrics against its labeled data. This proves the evaluation infrastructure works before
Wolf Archive labels exist.

This is not a new module — it is a test script that uses the pipeline and evaluation modules.

**Creates:** `tests/test_covid_eval.py`

```python
"""COVID/Poynter retrieval evaluation.

Exercises the full pipeline on labeled data:
- Ingest COVID dataset into ChromaDB
- Run sample queries
- Compute retrieval metrics against known labels
- Verify SQLite logging
"""
import polars as pl
import pytest
from promptrag.pipeline import RAGPipeline, load_config
from promptrag.evaluation import precision_at_k, recall_at_k, ndcg_at_k, mrr


@pytest.fixture(scope="module")
def pipeline():
    config = load_config("configs/covid_poynter.yaml")
    pipe = RAGPipeline(config)
    yield pipe
    pipe.close()


def test_ingest(pipeline):
    """Verify COVID data ingests into ChromaDB."""
    count = pipeline.ingest()
    assert count > 0
    assert pipeline.collection.count() == count


def test_query_returns_results(pipeline):
    """Run a query and verify results come back with expected structure."""
    result = pipeline.query("COVID vaccine causes infertility")
    assert result["run_id"] is not None
    assert len(result["retrieved_docs"]) > 0
    assert len(result["scores"]) == len(result["retrieved_docs"])
    assert result["llm_output"]  # non-empty


def test_retrieval_metrics(pipeline):
    """Compute retrieval metrics using code labels as relevance."""
    config = load_config("configs/covid_poynter.yaml")
    df = pl.read_csv(
        config["dataset"]["path"],
        separator=config["dataset"].get("separator", ","),
    )

    # Pick a labeled example and use it as a query
    sample = df.row(0, named=True)
    query_text = sample[config["dataset"]["text_column"]]
    query_label = sample[config["dataset"]["label_column"]]

    # Get retrieval results
    results = pipeline.vector_store.query(
        pipeline.collection, [query_text], k=5
    )
    retrieved_ids = results["ids"][0]

    # Relevant = docs with the same code label
    relevant_ids = set(
        str(x) for x, code in zip(
            df[config["dataset"]["id_column"]].to_list(),
            df[config["dataset"]["label_column"]].to_list(),
        )
        if code == query_label
    )

    p = precision_at_k(retrieved_ids, relevant_ids, k=5)
    r = recall_at_k(retrieved_ids, relevant_ids, k=5)
    rr = mrr(retrieved_ids, relevant_ids)
    ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k=5)

    print(f"\nQuery label: {query_label}")
    print(f"  Precision@5: {p:.3f}")
    print(f"  Recall@5: {r:.3f}")
    print(f"  MRR: {rr:.3f}")
    print(f"  nDCG@5: {ndcg:.3f}")

    # Sanity — metrics should be computable (no crashes), values in [0, 1]
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
```

**Key design decisions:**
- Uses COVID/Poynter because it has labels (`code` column) — Wolf Archive does not yet.
- Retrieval relevance defined as "same misinformation code" — simple but exercises the full
  metrics pipeline.
- Wolf Archive benchmark test (pretend-play/conflict) deferred to Phase 2 when gold set labels
  are created. The test structure is documented in research.md and can be implemented once
  the 24 episodes are identified and labeled.

**Test point:** `pytest tests/test_covid_eval.py -v -s` shows retrieval metrics and verifies
SQLite logging.

---

## Implementation Order Summary

| Step | Module | Depends On | Test Point |
|---|---|---|---|
| 0.1 | Archive existing code | — | Branch exists |
| 0.2 | Dataset inventory | — | Both CSVs in `data/` confirmed |
| 1 | Package scaffolding | — | `import promptrag` works |
| 2 | YAML config | Step 0.2 (column names) | Config loads |
| 3 | Embedding module | Step 1 | Encode + dimension check |
| 4 | Chunker | Step 1 | Pass-through + split tests |
| 5 | Vector store | Steps 3, 4 | Add + query + persistence |
| 6 | LLM backend | Step 1 | Manual generation test |
| 7 | Prompt engine | Step 1 | Message structure test |
| 8 | Evaluation module | Step 1 | Metrics + SQLite logging |
| 9 | Pipeline orchestrator | Steps 3-8 | End-to-end ingest + query |
| 10 | CLI | Step 9 | `promptrag --help` |
| 11 | COVID eval run | Steps 9, 0.2 | Retrieval metrics on labeled data |

Steps 3, 4, 6, 7, 8 are independent of each other and can be implemented in parallel.
Steps 5, 9, 10, 11 are sequential.

---

## Checklist — Plan → Implement Phase Gate

- [x] Plan Doc exists at `docs/plan.md`
- [x] Each module change has exact file name
- [x] Each module change has a code snippet showing what changes
- [x] Wolf Archive benchmark acknowledged — deferred to Phase 2 pending labeling
- [x] COVID/Poynter retrieval eval is the Phase 1 eval test point (Step 11)
- [x] ChromaDB collection schema matches naming convention (`wolf_archive_modernbert_base`)
- [x] SQLite run record schema matches research.md Decision 4 (13 columns)
- [x] MPS device configuration is explicit (Steps 3, 6)
- [x] Scope exclusions are listed (top of document)
