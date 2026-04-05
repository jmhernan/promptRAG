# promptRAG

Research-team-grade RAG pipeline for institutional and administrative text. Built for rigorous, reproducible evaluation of retrieval and generation quality — not a wrapper around GPT-4.

## What This Is

promptRAG is a clean, modular RAG pipeline designed for small research teams working with messy, high-stakes text: court records, field notes, policy documents. It treats **evaluation as a first-class concern** — every run is logged to SQLite with full provenance, enabling apples-to-apples comparisons across embedding models, LLMs, and retrieval configurations.

The research focus is on **small language models (SLMs) in retrieval-augmented settings** — understanding where Phi, Qwen, Gemma, and Mistral-class models fail, and whether failures stem from retrieval or generation.

### Origin

The semantic search approach in this pipeline was first published in:

> Xu, J. & Hernandez, J. M. (2025). "Reading children's moral dramas in anthropological fieldnotes: A human–AI hybrid approach." *Cambridge Forum on AI: Culture and Society*, 1, e6. doi:10.1017/cfc.2025.10008

## Pipeline

```
Data (CSV)
  │
  ▼  chunker.py
Chunk Documents (fixed-size, with pass-through for short docs)
  │
  ▼  embeddings.py
Encode with SentenceTransformer (nomic-ai/modernbert-embed-base)
  │
  ▼  vector_store.py
Store in ChromaDB (persistent, metadata-filtered)
  │
  ▼  prompt_engine.py
Build Chat Messages (Jinja2 templates)
  │
  ▼  llm_backend.py
Generate with HuggingFace transformers (apply_chat_template)
  │
  ▼  evaluation.py
Log Run to SQLite + Compute Retrieval Metrics
```

## Project Structure

```
src/promptrag/
  __init__.py
  embeddings.py       # SentenceTransformer wrapper — MPS/CUDA/CPU, batch encoding, query prefixes
  chunker.py          # Fixed-size word-count chunking with overlap
  vector_store.py     # ChromaDB persistent client — add, query, collection management
  llm_backend.py      # HuggingFace transformers text-generation pipeline
  prompt_engine.py    # Jinja2 chat message builder
  evaluation.py       # Retrieval metrics (P@k, R@k, MRR, nDCG@k) + SQLite run logger
  pipeline.py         # End-to-end orchestrator — ingest, query, log
  cli.py              # Click CLI — promptrag ingest / query
configs/
  wolf_archive.yaml       # Wolf Archive field notes config
  covid_poynter.yaml      # COVID/Poynter misinformation config
scripts/
  test_retrieval.py       # Retrieval-only test (no LLM needed)
  test_full_pipeline.py   # Full pipeline test (requires LLM download)
tests/
  test_embeddings.py
  test_chunker.py
  test_vector_store.py
  test_evaluation.py
docs/
  research.md         # Research document — codebase audit, literature, decisions
  plan.md             # Implementation plan
  decision-log.md     # Real-time decision log
experiments/          # Runtime artifacts — ChromaDB, SQLite (gitignored)
```

## Usage

### CLI

```bash
# Ingest a dataset into ChromaDB
promptrag ingest --config configs/wolf_archive.yaml

# Run a query
promptrag query --config configs/wolf_archive.yaml "children playing cooperatively"
```

### Python

```python
from promptrag.pipeline import RAGPipeline, load_config

config = load_config("configs/wolf_archive.yaml")
pipeline = RAGPipeline(config)
pipeline.ingest()

result = pipeline.query("children fighting or in physical conflict")
print(result["llm_output"])
print(result["scores"])

pipeline.close()
```

### Retrieval Only (no LLM download)

```python
from promptrag.embeddings import EmbeddingModel
from promptrag.vector_store import VectorStore
from promptrag.chunker import chunk_documents
import polars as pl

model = EmbeddingModel(device="mps")
store = VectorStore(model)
coll = store.get_or_create_collection("wolf_archive_modernbert_embed")

df = pl.read_csv("data/4ChildObservation_MasterFile.csv")
chunks = chunk_documents(df["text"].to_list(), [str(x) for x in df["Index"].to_list()])
store.add_chunks(coll, chunks)

results = store.query(coll, ["children playing a pretend war game"], k=5)
```

## Setup

```bash
# Python 3.11+ required
pip install -e ".[dev]"
```

### Key Dependencies

- **Embeddings**: `sentence-transformers`, `nomic-ai/modernbert-embed-base`
- **Vector Store**: `chromadb`
- **LLM**: `transformers` (HuggingFace pipelines)
- **Data**: `polars`
- **Evaluation**: `sqlite3` (stdlib)
- **CLI**: `click`
- **Templates**: `jinja2`

### Hardware

Designed for Apple Silicon (M4 Max). Uses MPS backend with automatic fallback to CPU. CUDA supported where available.

## Datasets

| Dataset | Text Type | Task | Status |
|---|---|---|---|
| Wolf Archive | Short field notes (~300 words) | Semantic theme classification | Active |
| COVID/Poynter | Short tweets | Misinformation classification | Active (eval metrics) |
| Eviction Records | Long legal text | Targeted extraction | Phase 2 |

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vector store | ChromaDB | Persistence + metadata filtering built in |
| Embedding model | `nomic-ai/modernbert-embed-base` | Fine-tuned ModernBERT for retrieval; MTEB 62.6 |
| Evaluation logging | SQLite + Polars | Queryable, local, no infrastructure |
| LLM backend | HuggingFace `transformers` | Model internals access for SLM eval |
| DataFrame library | Polars | Fast, lazy eval, reads SQLite directly |

See [docs/research.md](docs/research.md) for full rationale and [docs/decision-log.md](docs/decision-log.md) for the audit trail.

## License

MIT — see [LICENSE](LICENSE).

## Additional: Transformer Playground

`src/transformer_playground.py` is a standalone educational implementation of transformer components (multi-head attention, feed-forward layers, encoder stack) with attention visualization. It is **not** part of the main RAG pipeline.
