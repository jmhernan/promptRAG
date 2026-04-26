# promptRAG — Phase 2 Plan Document: Classification

*Classification-via-RAG for nuanced institutional text.*
*Reference: `docs/research.md` Section 7 (April 2026), Phase 1 plan at `docs/plan.md`*

---

## Scope

### What is in Phase 2

- Label plumbing: read `label_column` from config, propagate through chunker → ChromaDB metadata
- Classification module (`src/promptrag/classifier.py`) with three strategies:
  1. kNN voting over retrieved neighbor labels
  2. Anchor-vector scoring (zero-shot, softmax distribution)
  3. RAG-augmented classification (few-shot in-context + LLM generation)
- Classification evaluation metrics in `evaluation.py`: accuracy, macro-F1, confusion matrix,
  per-category precision/recall
- COVID/Poynter as the primary validation dataset (19 categories, 317 rows, labeled)
- CLI commands for classification: `classify` (single doc) and `classify-eval` (batch evaluation)
- SQLite classification logging — extends the existing `runs` table or adds a `classification_runs` table
- Config schema extensions for classification task parameters

### What is NOT in Phase 2

- Wolf Archive gold set labeling (requires manual identification of the 24 pretend-play episodes)
- Eviction records dataset wiring
- Cross-encoder reranking
- Fine-tuning of any models
- Ollama / vLLM / API backends
- Hyperparameter search (k, temperature, prompt variants)
- Sentence-based or semantic chunking
- CI pipeline

### Dependency on Phase 1

Phase 2 assumes the following Phase 1 modules are working and tested:

| Module | Required API | Status |
|---|---|---|
| `embeddings.py` | `EmbeddingModel.encode(texts, is_query)` | ✓ Implemented |
| `vector_store.py` | `VectorStore.add_chunks()`, `.query()` | ✓ Implemented |
| `chunker.py` | `chunk_documents()` → `list[Chunk]` with metadata | ✓ Implemented |
| `llm_backend.py` | `HuggingFaceLLM.generate_from_messages()` | ✓ Implemented |
| `prompt_engine.py` | `PromptEngine.build_messages()` | ✓ Implemented |
| `evaluation.py` | `RunLogger.log_run()`, retrieval metrics | ✓ Implemented |
| `pipeline.py` | `RAGPipeline.ingest()`, `.query()` | ✓ Implemented |

---

## Pre-Implementation: Data Audit

### COVID/Poynter label distribution (validated April 2026)

```
 51  conspiracy
 42  false attribution
 30  virus downplaying
 29  vaccine
 27  cure
 24  government intervention
 18  medical supplies
 17  trump
 13  government covid funds
  9  virus trajectory abroad
  8  economy
  8  other outbreaks
  7  health authority
  7  virus origin
  6  virus transmission
  6  covid testing
  6  5g
  6  congress
  3  celebrities with covid

Total: 317 rows, 19 categories
```

**Note — separator issue:** The file is named `_tab.csv` but is actually **comma-separated**
(verified via byte inspection — no `\t` characters in header). The config currently says
`separator: "\t"`. This must be fixed. The pipeline's `pl.read_csv()` with `separator="\t"`
would produce a single-column DataFrame, silently breaking ingestion.

**Evaluation consideration:** 19 categories with a minimum of 3 samples (`celebrities with covid`)
means per-category precision/recall will be noisy for rare categories. Macro-F1 is the right
primary metric — it penalizes poor performance on minority categories without requiring
stratified sampling.

---

## Implementation Plan

### Step 0: Fix COVID config separator bug

**Goal:** Fix the incorrect `separator: "\t"` in `configs/covid_poynter.yaml`.

**File:** `configs/covid_poynter.yaml` line 6

**Change:**
```yaml
# FROM:
  separator: "\t"                    # tab-separated

# TO:
  separator: ","                     # comma-separated (file misnamed as _tab)
```

**Test point:** `python3 -c "import polars as pl; df = pl.read_csv('data/poynter_coded_breon_tab.csv', separator=','); print(len(df.columns), 'columns,', len(df), 'rows')"` → should print `18 columns, 317 rows` (not `1 column`).

---

### Step 1: Label plumbing — pipeline.ingest() reads labels into metadata

**Goal:** When `label_column` is present in the dataset config, read labels from the CSV
and propagate them through chunker metadata into ChromaDB.

**File:** `src/promptrag/pipeline.py`, method `ingest()` (lines 47–67)

**Current code (lines 53–67):**
```python
        texts = df[ds_config["text_column"]].to_list()
        ids = [str(x) for x in df[ds_config["id_column"]].to_list()]

        chunks = chunk_documents(
            texts=texts,
            doc_ids=ids,
            chunk_size=self.config["chunking"]["chunk_size"],
            chunk_overlap=self.config["chunking"]["chunk_overlap"],
        )
```

**New code:**
```python
        texts = df[ds_config["text_column"]].to_list()
        ids = [str(x) for x in df[ds_config["id_column"]].to_list()]

        # If label_column is configured, read labels and attach as metadata
        metadata = None
        label_column = ds_config.get("label_column")
        if label_column and label_column in df.columns:
            labels = df[label_column].to_list()
            metadata = [{"label": str(lbl) if lbl is not None else ""} for lbl in labels]

        chunks = chunk_documents(
            texts=texts,
            doc_ids=ids,
            chunk_size=self.config["chunking"]["chunk_size"],
            chunk_overlap=self.config["chunking"]["chunk_overlap"],
            metadata=metadata,
        )
```

**What this enables:** After re-ingestion, ChromaDB metadata for each chunk will include
`{"doc_id": "727", "chunk_index": 0, "label": "medical supplies"}`. The label is now
queryable via ChromaDB metadata filtering and available to the classifier.

**No changes needed in:**
- `chunker.py` — already accepts `metadata: list[dict] | None` and passes it through
- `vector_store.py` — already stores `{**c.metadata}` in ChromaDB metadatas

**Test point:**
```python
def test_ingest_stores_labels(tmp_path):
    """Labels from CSV flow through to ChromaDB metadata."""
    # Create a small test CSV
    import polars as pl
    df = pl.DataFrame({"id": [1, 2], "text": ["cat", "dog"], "label": ["A", "B"]})
    csv_path = tmp_path / "test.csv"
    df.write_csv(csv_path)

    config = {
        "dataset": {"path": str(csv_path), "text_column": "text",
                     "id_column": "id", "label_column": "label", "name": "test"},
        "embedding": {"model": "all-MiniLM-L6-v2", "device": "cpu", "batch_size": 32},
        "chunking": {"chunk_size": 512, "chunk_overlap": 50, "strategy": "fixed"},
        "vector_store": {"persist_directory": str(tmp_path / "chroma"),
                         "collection_name": "test_labels"},
        "retrieval": {"k": 2},
        "llm": {"model": "microsoft/Phi-3-mini-4k-instruct",
                "device_map": "cpu", "max_new_tokens": 10},
        "evaluation": {"db_path": str(tmp_path / "runs.db")},
    }
    # Only test ingest, not LLM — skip LLM init
    from promptrag.embeddings import EmbeddingModel
    from promptrag.vector_store import VectorStore
    from promptrag.chunker import chunk_documents
    import polars as pl

    ds_config = config["dataset"]
    df = pl.read_csv(ds_config["path"])
    texts = df["text"].to_list()
    ids = [str(x) for x in df["id"].to_list()]
    labels = df["label"].to_list()
    metadata = [{"label": str(lbl)} for lbl in labels]

    em = EmbeddingModel("all-MiniLM-L6-v2", device="cpu", query_prefix="", document_prefix="")
    vs = VectorStore(em, persist_directory=str(tmp_path / "chroma"))
    coll = vs.get_or_create_collection("test_labels")
    chunks = chunk_documents(texts, ids, metadata=metadata)
    vs.add_chunks(coll, chunks)

    # Verify metadata stored
    result = coll.get(ids=["1_chunk0"], include=["metadatas"])
    assert result["metadatas"][0]["label"] == "A"
```

---

### Step 2: Classification metrics in evaluation.py

**Goal:** Add classification-specific metrics alongside existing retrieval metrics.

**File:** `src/promptrag/evaluation.py` — append after the retrieval metrics section (after line 46)

**New code:**
```python
# --- Classification metrics ---

from collections import Counter


def accuracy(predicted: list[str], actual: list[str]) -> float:
    """Overall classification accuracy."""
    if not actual:
        return 0.0
    return sum(p == a for p, a in zip(predicted, actual)) / len(actual)


def per_category_precision_recall(
    predicted: list[str], actual: list[str]
) -> dict[str, dict[str, float]]:
    """Per-category precision and recall.

    Returns: {category: {"precision": float, "recall": float, "f1": float, "support": int}}
    """
    categories = sorted(set(actual) | set(predicted))
    results = {}
    for cat in categories:
        tp = sum(p == cat and a == cat for p, a in zip(predicted, actual))
        fp = sum(p == cat and a != cat for p, a in zip(predicted, actual))
        fn = sum(p != cat and a == cat for p, a in zip(predicted, actual))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = sum(1 for a in actual if a == cat)
        results[cat] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return results


def macro_f1(predicted: list[str], actual: list[str]) -> float:
    """Macro-averaged F1: unweighted mean of per-category F1 scores."""
    per_cat = per_category_precision_recall(predicted, actual)
    f1_scores = [v["f1"] for v in per_cat.values() if v["support"] > 0]
    if not f1_scores:
        return 0.0
    return sum(f1_scores) / len(f1_scores)


def confusion_matrix(
    predicted: list[str], actual: list[str]
) -> dict[str, dict[str, int]]:
    """Confusion matrix as nested dict: {actual_label: {predicted_label: count}}.

    Rows are actual, columns are predicted.
    """
    categories = sorted(set(actual) | set(predicted))
    matrix = {a: {p: 0 for p in categories} for a in categories}
    for p, a in zip(predicted, actual):
        matrix[a][p] += 1
    return matrix
```

**Test point (`tests/test_evaluation.py` — append):**
```python
from promptrag.evaluation import accuracy, macro_f1, per_category_precision_recall, confusion_matrix


def test_accuracy():
    assert accuracy(["A", "B", "A"], ["A", "B", "B"]) == pytest.approx(2 / 3)
    assert accuracy([], []) == 0.0


def test_macro_f1_perfect():
    pred = ["A", "A", "B", "B"]
    actual = ["A", "A", "B", "B"]
    assert macro_f1(pred, actual) == 1.0


def test_macro_f1_partial():
    pred = ["A", "A", "B", "A"]
    actual = ["A", "B", "B", "A"]
    score = macro_f1(pred, actual)
    assert 0 < score < 1.0


def test_per_category_precision_recall():
    pred = ["A", "A", "B", "B"]
    actual = ["A", "B", "B", "B"]
    result = per_category_precision_recall(pred, actual)
    assert result["A"]["precision"] == 0.5  # 1 TP, 1 FP
    assert result["A"]["recall"] == 1.0     # 1 TP, 0 FN
    assert result["B"]["precision"] == 1.0  # 2 TP, 0 FP
    assert result["B"]["recall"] == pytest.approx(2 / 3)  # 2 TP, 1 FN


def test_confusion_matrix():
    pred = ["A", "B", "A"]
    actual = ["A", "B", "B"]
    cm = confusion_matrix(pred, actual)
    assert cm["A"]["A"] == 1
    assert cm["B"]["A"] == 1  # misclassified B as A
    assert cm["B"]["B"] == 1
```

---

### Step 3: Classifier module — kNN voting (Strategy 1)

**Goal:** Create `src/promptrag/classifier.py` with the simplest classification strategy.

**Creates:** `src/promptrag/classifier.py`

```python
"""Classification strategies built on the promptRAG retrieval layer."""

from collections import Counter
from promptrag.vector_store import VectorStore


def knn_classify(
    vector_store: VectorStore,
    collection,
    query_text: str,
    k: int = 5,
    weight_by_score: bool = False,
) -> dict:
    """Classify a document by majority vote over labels of k nearest neighbors.

    Args:
        vector_store: Initialized VectorStore with embedded corpus.
        collection: ChromaDB collection to query.
        query_text: Text to classify.
        k: Number of neighbors to retrieve.
        weight_by_score: If True, weight votes by similarity score.

    Returns:
        {
            "predicted_label": str,
            "label_scores": dict[str, float],  # vote tallies (or weighted scores)
            "retrieved_ids": list[str],
            "retrieved_labels": list[str],
            "similarity_scores": list[float],
        }
    """
    results = vector_store.query(collection, [query_text], k=k)
    retrieved_ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    scores = [1.0 - d for d in distances]

    retrieved_labels = [m.get("label", "") for m in metadatas]

    if weight_by_score:
        label_scores: dict[str, float] = {}
        for label, score in zip(retrieved_labels, scores):
            if label:
                label_scores[label] = label_scores.get(label, 0.0) + score
    else:
        counts = Counter(lbl for lbl in retrieved_labels if lbl)
        label_scores = {lbl: float(n) for lbl, n in counts.items()}

    predicted_label = max(label_scores, key=label_scores.get) if label_scores else ""

    return {
        "predicted_label": predicted_label,
        "label_scores": label_scores,
        "retrieved_ids": retrieved_ids,
        "retrieved_labels": retrieved_labels,
        "similarity_scores": scores,
    }


def knn_classify_batch(
    vector_store: VectorStore,
    collection,
    texts: list[str],
    k: int = 5,
    weight_by_score: bool = False,
) -> list[dict]:
    """Classify a batch of documents via kNN voting.

    Calls knn_classify per document. Returns list of result dicts.
    """
    return [
        knn_classify(vector_store, collection, text, k=k, weight_by_score=weight_by_score)
        for text in texts
    ]
```

**Design decisions:**
- `knn_classify` returns a full result dict (not just the label) so the evaluation layer
  can inspect retrieved neighbors, scores, and vote breakdown.
- `weight_by_score=False` is the default — unweighted majority vote. Weighted variant
  available for experimentation.
- Batch function is a simple loop. ChromaDB supports batch queries, but per-query results
  are simpler to reason about and log individually.
- The function reads labels from ChromaDB metadata (`m.get("label", "")`) — this is why
  Step 1 (label plumbing) must come first.

**Test point (`tests/test_classifier.py` — new file):**
```python
import pytest
from promptrag.embeddings import EmbeddingModel
from promptrag.vector_store import VectorStore
from promptrag.chunker import chunk_documents, Chunk
from promptrag.classifier import knn_classify


@pytest.fixture
def labeled_collection(tmp_path):
    """Small labeled corpus in ChromaDB for classification tests."""
    em = EmbeddingModel("all-MiniLM-L6-v2", device="cpu", query_prefix="", document_prefix="")
    vs = VectorStore(em, persist_directory=str(tmp_path / "chroma"))
    coll = vs.get_or_create_collection("test_classify")
    chunks = chunk_documents(
        texts=["the cat sat on the mat", "dogs love to play fetch",
               "a kitten sleeping on a pillow", "the puppy chased the ball"],
        doc_ids=["d1", "d2", "d3", "d4"],
        metadata=[{"label": "cat"}, {"label": "dog"}, {"label": "cat"}, {"label": "dog"}],
    )
    vs.add_chunks(coll, chunks)
    return vs, coll


def test_knn_classify_returns_label(labeled_collection):
    vs, coll = labeled_collection
    result = knn_classify(vs, coll, "a feline resting quietly", k=4)
    assert result["predicted_label"] in ("cat", "dog")
    assert len(result["retrieved_ids"]) == 4
    assert len(result["retrieved_labels"]) == 4


def test_knn_classify_weighted(labeled_collection):
    vs, coll = labeled_collection
    result = knn_classify(vs, coll, "a feline resting quietly", k=4, weight_by_score=True)
    assert "label_scores" in result
    assert all(isinstance(v, float) for v in result["label_scores"].values())
```

---

### Step 4: Classifier module — anchor-vector scoring (Strategy 2)

**Goal:** Add anchor-vector classification to `classifier.py`. This generalizes the
Wolf Archive theme-vector approach.

**Appends to:** `src/promptrag/classifier.py`

**New code:**
```python
import numpy as np
from promptrag.embeddings import EmbeddingModel


def anchor_classify(
    embedding_model: EmbeddingModel,
    document_text: str,
    anchors: dict[str, str],
    temperature: float = 1.0,
) -> dict:
    """Classify by scoring a document against fixed anchor descriptions.

    Args:
        embedding_model: Initialized EmbeddingModel.
        document_text: Text to classify.
        anchors: {label: description_text} — category descriptions as anchor vectors.
        temperature: Softmax temperature. Lower = sharper distribution.

    Returns:
        {
            "predicted_label": str,
            "label_scores": dict[str, float],  # softmax probabilities over anchors
            "raw_similarities": dict[str, float],  # cosine similarities before softmax
        }
    """
    labels = list(anchors.keys())
    descriptions = list(anchors.values())

    # Encode anchors as documents (they are category descriptions, not queries)
    anchor_embeddings = np.array(
        embedding_model.encode(descriptions, is_query=False)
    )
    # Encode the document
    doc_embedding = np.array(
        embedding_model.encode([document_text], is_query=False)
    )[0]

    # Cosine similarity (embeddings are already normalized by SentenceTransformer)
    similarities = anchor_embeddings @ doc_embedding

    # Softmax with temperature
    scaled = similarities / temperature
    exp_scores = np.exp(scaled - np.max(scaled))  # numerical stability
    softmax_scores = exp_scores / exp_scores.sum()

    raw_sims = {label: float(sim) for label, sim in zip(labels, similarities)}
    label_probs = {label: float(prob) for label, prob in zip(labels, softmax_scores)}
    predicted = max(label_probs, key=label_probs.get)

    return {
        "predicted_label": predicted,
        "label_scores": label_probs,
        "raw_similarities": raw_sims,
    }


def anchor_classify_batch(
    embedding_model: EmbeddingModel,
    texts: list[str],
    anchors: dict[str, str],
    temperature: float = 1.0,
) -> list[dict]:
    """Classify a batch of documents against anchor vectors.

    Optimized: encodes all documents in one batch, computes all similarities at once.
    """
    labels = list(anchors.keys())
    descriptions = list(anchors.values())

    anchor_embeddings = np.array(
        embedding_model.encode(descriptions, is_query=False)
    )
    doc_embeddings = np.array(
        embedding_model.encode(texts, is_query=False)
    )

    # All similarities at once: (n_docs, n_anchors)
    sim_matrix = doc_embeddings @ anchor_embeddings.T

    results = []
    for i, text in enumerate(texts):
        similarities = sim_matrix[i]
        scaled = similarities / temperature
        exp_scores = np.exp(scaled - np.max(scaled))
        softmax_scores = exp_scores / exp_scores.sum()

        raw_sims = {label: float(sim) for label, sim in zip(labels, similarities)}
        label_probs = {label: float(prob) for label, prob in zip(labels, softmax_scores)}
        predicted = max(label_probs, key=label_probs.get)

        results.append({
            "predicted_label": predicted,
            "label_scores": label_probs,
            "raw_similarities": raw_sims,
        })
    return results
```

**Design decisions:**
- `anchors` is a simple `dict[str, str]` — label to description. The caller constructs
  these from domain knowledge, topic modeling, or keyword lists.
- `temperature` parameter controls the sharpness of the softmax distribution. At `temperature=1.0`,
  the distribution is relatively flat. Lower values make the classification more decisive.
  This is an experiment variable.
- Batch version encodes all documents once and computes the full similarity matrix, avoiding
  repeated embedding calls.
- Anchor embeddings use `is_query=False` (document prefix) — anchors are category descriptions,
  not search queries. This matches the Wolf Archive design where themes were encoded as
  composite descriptions.

**Test point (append to `tests/test_classifier.py`):**
```python
from promptrag.classifier import anchor_classify, anchor_classify_batch


def test_anchor_classify():
    em = EmbeddingModel("all-MiniLM-L6-v2", device="cpu", query_prefix="", document_prefix="")
    anchors = {
        "animal": "cats dogs pets animals wildlife",
        "vehicle": "cars trucks buses trains transportation",
    }
    result = anchor_classify(em, "the kitten purred on the sofa", anchors)
    assert result["predicted_label"] == "animal"
    assert abs(sum(result["label_scores"].values()) - 1.0) < 1e-6  # softmax sums to 1


def test_anchor_classify_batch():
    em = EmbeddingModel("all-MiniLM-L6-v2", device="cpu", query_prefix="", document_prefix="")
    anchors = {
        "animal": "cats dogs pets animals",
        "vehicle": "cars trucks buses trains",
    }
    texts = ["the kitten purred", "the truck drove fast"]
    results = anchor_classify_batch(em, texts, anchors)
    assert len(results) == 2
    assert results[0]["predicted_label"] == "animal"
    assert results[1]["predicted_label"] == "vehicle"
```

---

### Step 5: Classifier module — RAG-augmented classification (Strategy 3)

**Goal:** Add LLM-based classification to `classifier.py`. Retrieves labeled examples
as in-context demonstrations, then prompts the LLM to classify.

**Appends to:** `src/promptrag/classifier.py`

**New code:**
```python
from promptrag.llm_backend import HuggingFaceLLM
from promptrag.prompt_engine import PromptEngine


CLASSIFICATION_SYSTEM_PROMPT = (
    "You are a document classifier. Based on the labeled examples provided "
    "as context, classify the target document into exactly one of the categories "
    "shown. Reply with ONLY the category label, nothing else."
)

CLASSIFICATION_USER_TEMPLATE = (
    "Labeled examples:\n"
    "{% for doc, label in examples %}"
    '- [{{ label }}]: {{ doc }}\n'
    "{% endfor %}\n\n"
    "Valid categories: {{ categories }}\n\n"
    "Classify this document: {{ query }}"
)


def rag_classify(
    vector_store: VectorStore,
    collection,
    llm: HuggingFaceLLM,
    query_text: str,
    valid_labels: list[str],
    k: int = 5,
    system_prompt: str = CLASSIFICATION_SYSTEM_PROMPT,
    user_template: str = CLASSIFICATION_USER_TEMPLATE,
) -> dict:
    """Classify a document using retrieved labeled examples + LLM reasoning.

    Args:
        vector_store: Initialized VectorStore.
        collection: ChromaDB collection with labeled documents.
        llm: Initialized HuggingFaceLLM.
        query_text: Text to classify.
        valid_labels: List of valid category labels (constrains LLM output).
        k: Number of labeled examples to retrieve.
        system_prompt: System prompt for classification.
        user_template: Jinja2 template for user message.

    Returns:
        {
            "predicted_label": str,       # parsed from LLM output
            "raw_output": str,            # full LLM output text
            "retrieved_ids": list[str],
            "retrieved_labels": list[str],
            "similarity_scores": list[float],
        }
    """
    # Retrieve labeled neighbors
    results = vector_store.query(collection, [query_text], k=k)
    retrieved_ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    retrieved_docs = results["documents"][0]
    scores = [1.0 - d for d in distances]
    retrieved_labels = [m.get("label", "") for m in metadatas]

    # Build classification prompt
    examples = list(zip(retrieved_docs, retrieved_labels))
    categories_str = ", ".join(valid_labels)

    prompt_engine = PromptEngine(
        system_prompt=system_prompt,
        user_template=user_template,
    )
    messages = prompt_engine.build_messages(
        query=query_text,
        retrieved_docs=[],  # not used — template uses examples directly
    )
    # Override user content with classification-specific rendering
    from jinja2 import Environment, BaseLoader
    env = Environment(loader=BaseLoader())
    template = env.from_string(user_template)
    user_content = template.render(
        examples=examples,
        categories=categories_str,
        query=query_text,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw_output = llm.generate_from_messages(messages)

    # Parse label from LLM output — match against valid labels
    predicted_label = _parse_label(raw_output.strip(), valid_labels)

    return {
        "predicted_label": predicted_label,
        "raw_output": raw_output,
        "retrieved_ids": retrieved_ids,
        "retrieved_labels": retrieved_labels,
        "similarity_scores": scores,
    }


def _parse_label(llm_output: str, valid_labels: list[str]) -> str:
    """Extract a valid label from LLM output.

    Strategy: check if any valid label appears in the output (case-insensitive).
    Prefer exact match, then substring match, then return raw output.
    """
    output_lower = llm_output.lower().strip()

    # Exact match
    for label in valid_labels:
        if output_lower == label.lower():
            return label

    # Substring match — first valid label found in output
    for label in valid_labels:
        if label.lower() in output_lower:
            return label

    # No match — return raw output (will be counted as misclassification)
    return llm_output.strip()
```

**Design decisions:**
- `valid_labels` is required — constrains the label space and enables `_parse_label()`
  to extract structured output from potentially noisy LLM text.
- `_parse_label()` is deliberately simple: exact match → substring match → raw output.
  More sophisticated parsing (regex, constrained decoding) can be added later if needed.
  If the raw output is returned, it will not match any valid label and will be counted as
  a misclassification in evaluation — this is intentional. It exposes LLM failure modes.
- The system prompt is explicit: "Reply with ONLY the category label, nothing else."
  This works well with instruction-tuned models (Phi-3, Qwen, etc.).
- `user_template` includes the valid categories list — this primes the LLM and reduces
  hallucinated labels.
- The function builds messages directly rather than using the generic `PromptEngine.build_messages()`
  because the classification template has different variables (`examples`, `categories`)
  than the retrieval QA template (`retrieved_docs`, `query`).

**Test point:** Unit-testing the RAG classifier requires an LLM, which is expensive.
Test `_parse_label()` independently:
```python
from promptrag.classifier import _parse_label


def test_parse_label_exact():
    assert _parse_label("conspiracy", ["conspiracy", "vaccine"]) == "conspiracy"


def test_parse_label_case_insensitive():
    assert _parse_label("Conspiracy", ["conspiracy", "vaccine"]) == "conspiracy"


def test_parse_label_substring():
    assert _parse_label("I think this is a conspiracy theory", ["conspiracy", "vaccine"]) == "conspiracy"


def test_parse_label_no_match():
    result = _parse_label("I don't know", ["conspiracy", "vaccine"])
    assert result == "I don't know"
```

---

### Step 6: Classification logging

**Goal:** Add a classification-specific logging table to SQLite alongside the existing
`runs` table.

**File:** `src/promptrag/evaluation.py` — extend `RunLogger`

**New table schema:**
```python
CREATE_CLASSIFICATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS classification_runs (
    run_id              TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    dataset             TEXT NOT NULL,
    embedding_model     TEXT NOT NULL,
    strategy            TEXT NOT NULL,
    llm_model           TEXT,
    k                   INTEGER,
    doc_id              TEXT NOT NULL,
    doc_text            TEXT NOT NULL,
    actual_label        TEXT NOT NULL,
    predicted_label     TEXT NOT NULL,
    label_scores        TEXT,
    retrieved_ids       TEXT,
    retrieved_labels    TEXT,
    similarity_scores   TEXT,
    raw_llm_output      TEXT,
    notes               TEXT
)
"""
```

**New method on RunLogger:**
```python
    def log_classification(
        self,
        dataset: str,
        embedding_model: str,
        strategy: str,
        doc_id: str,
        doc_text: str,
        actual_label: str,
        predicted_label: str,
        llm_model: str = "",
        k: int | None = None,
        label_scores: dict | None = None,
        retrieved_ids: list[str] | None = None,
        retrieved_labels: list[str] | None = None,
        similarity_scores: list[float] | None = None,
        raw_llm_output: str = "",
        notes: str = "",
    ) -> str:
        run_id = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO classification_runs
            (run_id, timestamp, dataset, embedding_model, strategy, llm_model, k,
             doc_id, doc_text, actual_label, predicted_label, label_scores,
             retrieved_ids, retrieved_labels, similarity_scores, raw_llm_output, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                dataset,
                embedding_model,
                strategy,
                llm_model,
                k,
                doc_id,
                doc_text,
                actual_label,
                predicted_label,
                json.dumps(label_scores) if label_scores else None,
                json.dumps(retrieved_ids) if retrieved_ids else None,
                json.dumps(retrieved_labels) if retrieved_labels else None,
                json.dumps(similarity_scores) if similarity_scores else None,
                raw_llm_output,
                notes,
            ),
        )
        self.conn.commit()
        return run_id
```

**Also update `__init__`** to create the new table:
```python
    def __init__(self, db_path: str = "experiments/runs.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(CREATE_TABLE_SQL)
        self.conn.execute(CREATE_CLASSIFICATION_TABLE_SQL)
        self.conn.commit()
```

**Design decisions:**
- Separate table (not extending `runs`) — classification runs have different columns
  (`actual_label`, `predicted_label`, `strategy`) and different cardinality (one row per
  document, not per query).
- `strategy` column enables filtering by classification method in cross-comparison queries:
  `SELECT strategy, AVG(predicted_label = actual_label) FROM classification_runs GROUP BY strategy`
- `llm_model` is nullable — kNN and anchor-vector strategies don't use an LLM.
- `doc_text` stored for post-hoc analysis of misclassifications — what did the model get wrong,
  and what does the text look like?

**Test point (append to `tests/test_evaluation.py`):**
```python
def test_classification_logger(tmp_path):
    logger = RunLogger(db_path=str(tmp_path / "test.db"))
    run_id = logger.log_classification(
        dataset="covid",
        embedding_model="modernbert_embed",
        strategy="knn",
        doc_id="727",
        doc_text="face masks result in dangerous oxygen levels",
        actual_label="medical supplies",
        predicted_label="medical supplies",
        k=5,
        label_scores={"medical supplies": 3.0, "conspiracy": 2.0},
        retrieved_ids=["728_chunk0", "729_chunk0"],
        retrieved_labels=["medical supplies", "conspiracy"],
        similarity_scores=[0.92, 0.88],
    )
    assert run_id is not None
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    row = conn.execute(
        "SELECT * FROM classification_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    conn.close()
    logger.close()
```

---

### Step 7: Config schema extensions

**Goal:** Add classification-specific config fields.

**File:** `configs/covid_poynter.yaml` — add classification section

```yaml
classification:
  strategy: knn                      # knn | anchor | rag
  k: 5                               # neighbors for knn and rag strategies
  weight_by_score: false              # knn: similarity-weighted voting
  temperature: 1.0                   # anchor: softmax temperature
  anchors: null                      # anchor: {label: description} mapping (or path to YAML)
  valid_labels: null                  # rag: auto-populated from label_column if null
```

**File:** `configs/wolf_archive.yaml` — add classification section with anchor example

```yaml
classification:
  strategy: anchor
  temperature: 1.0
  anchors:
    family: "family relationships, parents, siblings, household, domestic"
    school: "school, classroom, teacher, student, learning, education"
    play: "play, game, pretend, imagination, fun, toys"
    cooperation: "cooperation, sharing, helping, teamwork, working together"
    conflict: "conflict, fighting, arguing, hitting, aggression, anger"
    shopping: "shopping, market, buying, selling, goods, trade"
```

**Design decisions:**
- `valid_labels: null` auto-populates from unique values of `label_column` at runtime.
  This avoids manually listing 19 COVID categories in the config.
- `anchors` can be defined inline (as above) or loaded from a separate YAML file for
  complex anchor sets. Inline is sufficient for Phase 2.
- The Wolf Archive anchors above are a starting point based on the six themes from the
  published paper. They will need refinement based on evaluation results.

---

### Step 8: CLI commands for classification

**Goal:** Add `classify` and `classify-eval` CLI commands.

**File:** `src/promptrag/cli.py` — append new commands

```python
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--strategy", type=click.Choice(["knn", "anchor", "rag"]), default=None,
              help="Override strategy from config")
@click.option("--k", default=None, type=int)
@click.argument("text")
def classify(config, strategy, k, text):
    """Classify a single document."""
    cfg = load_config(config)
    strategy = strategy or cfg.get("classification", {}).get("strategy", "knn")

    from promptrag.embeddings import EmbeddingModel
    from promptrag.vector_store import VectorStore
    from promptrag.classifier import knn_classify, anchor_classify, rag_classify

    em = EmbeddingModel(
        model_name=cfg["embedding"]["model"],
        device=cfg["embedding"]["device"],
    )

    if strategy == "knn":
        vs = VectorStore(em, persist_directory=cfg["vector_store"]["persist_directory"])
        coll = vs.get_or_create_collection(cfg["vector_store"]["collection_name"])
        k = k or cfg.get("classification", {}).get("k", 5)
        weight = cfg.get("classification", {}).get("weight_by_score", False)
        result = knn_classify(vs, coll, text, k=k, weight_by_score=weight)

    elif strategy == "anchor":
        anchors = cfg["classification"]["anchors"]
        temp = cfg.get("classification", {}).get("temperature", 1.0)
        result = anchor_classify(em, text, anchors, temperature=temp)

    elif strategy == "rag":
        vs = VectorStore(em, persist_directory=cfg["vector_store"]["persist_directory"])
        coll = vs.get_or_create_collection(cfg["vector_store"]["collection_name"])
        from promptrag.llm_backend import HuggingFaceLLM
        llm = HuggingFaceLLM(
            model_name=cfg["llm"]["model"],
            device_map=cfg["llm"]["device_map"],
            max_new_tokens=cfg["llm"]["max_new_tokens"],
        )
        k = k or cfg.get("classification", {}).get("k", 5)
        # Auto-populate valid labels from data if not specified
        valid_labels = cfg.get("classification", {}).get("valid_labels")
        if not valid_labels:
            import polars as pl
            ds = cfg["dataset"]
            df = pl.read_csv(ds["path"], separator=ds.get("separator", ","))
            valid_labels = sorted(df[ds["label_column"]].drop_nulls().unique().to_list())
        result = rag_classify(vs, coll, llm, text, valid_labels=valid_labels, k=k)

    click.echo(f"\nPredicted: {result['predicted_label']}")
    click.echo(f"Scores: {result['label_scores']}")


@main.command("classify-eval")
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--strategy", type=click.Choice(["knn", "anchor", "rag"]), default=None)
@click.option("--k", default=None, type=int)
@click.option("--limit", default=None, type=int, help="Max documents to evaluate (for testing)")
def classify_eval(config, strategy, k, limit):
    """Run classification evaluation on the full labeled dataset."""
    cfg = load_config(config)
    strategy = strategy or cfg.get("classification", {}).get("strategy", "knn")

    import polars as pl
    from promptrag.embeddings import EmbeddingModel
    from promptrag.vector_store import VectorStore
    from promptrag.evaluation import (
        RunLogger, accuracy, macro_f1, per_category_precision_recall, confusion_matrix,
    )
    from promptrag.classifier import knn_classify, anchor_classify_batch, rag_classify

    ds = cfg["dataset"]
    label_col = ds.get("label_column")
    if not label_col:
        click.echo("Error: dataset config must have label_column for evaluation.", err=True)
        raise SystemExit(1)

    df = pl.read_csv(ds["path"], separator=ds.get("separator", ","))
    df = df.filter(pl.col(ds["text_column"]).is_not_null() & pl.col(label_col).is_not_null())
    if limit:
        df = df.head(limit)

    texts = df[ds["text_column"]].to_list()
    ids = [str(x) for x in df[ds["id_column"]].to_list()]
    actual_labels = [str(x) for x in df[label_col].to_list()]

    em = EmbeddingModel(model_name=cfg["embedding"]["model"], device=cfg["embedding"]["device"])
    logger = RunLogger(db_path=cfg["evaluation"]["db_path"])
    k = k or cfg.get("classification", {}).get("k", 5)

    predicted_labels = []

    if strategy == "knn":
        vs = VectorStore(em, persist_directory=cfg["vector_store"]["persist_directory"])
        coll = vs.get_or_create_collection(cfg["vector_store"]["collection_name"])
        for i, (text, doc_id, actual) in enumerate(zip(texts, ids, actual_labels)):
            result = knn_classify(vs, coll, text, k=k,
                                  weight_by_score=cfg.get("classification", {}).get("weight_by_score", False))
            predicted_labels.append(result["predicted_label"])
            logger.log_classification(
                dataset=ds["name"], embedding_model=cfg["embedding"]["model"],
                strategy="knn", doc_id=doc_id, doc_text=text,
                actual_label=actual, predicted_label=result["predicted_label"],
                k=k, label_scores=result["label_scores"],
                retrieved_ids=result["retrieved_ids"],
                retrieved_labels=result["retrieved_labels"],
                similarity_scores=result["similarity_scores"],
            )
            if (i + 1) % 50 == 0:
                click.echo(f"  Classified {i + 1}/{len(texts)}...")

    elif strategy == "anchor":
        anchors = cfg["classification"]["anchors"]
        temp = cfg.get("classification", {}).get("temperature", 1.0)
        results = anchor_classify_batch(em, texts, anchors, temperature=temp)
        for doc_id, actual, result in zip(ids, actual_labels, results):
            predicted_labels.append(result["predicted_label"])
            logger.log_classification(
                dataset=ds["name"], embedding_model=cfg["embedding"]["model"],
                strategy="anchor", doc_id=doc_id,
                doc_text="",  # skip storing full text for batch — queryable from source
                actual_label=actual, predicted_label=result["predicted_label"],
                label_scores=result["label_scores"],
            )

    elif strategy == "rag":
        vs = VectorStore(em, persist_directory=cfg["vector_store"]["persist_directory"])
        coll = vs.get_or_create_collection(cfg["vector_store"]["collection_name"])
        from promptrag.llm_backend import HuggingFaceLLM
        llm = HuggingFaceLLM(
            model_name=cfg["llm"]["model"], device_map=cfg["llm"]["device_map"],
            max_new_tokens=cfg["llm"]["max_new_tokens"],
        )
        valid_labels = sorted(set(actual_labels))
        for i, (text, doc_id, actual) in enumerate(zip(texts, ids, actual_labels)):
            result = rag_classify(vs, coll, llm, text, valid_labels=valid_labels, k=k)
            predicted_labels.append(result["predicted_label"])
            logger.log_classification(
                dataset=ds["name"], embedding_model=cfg["embedding"]["model"],
                strategy="rag", llm_model=cfg["llm"]["model"],
                doc_id=doc_id, doc_text=text,
                actual_label=actual, predicted_label=result["predicted_label"],
                k=k, retrieved_ids=result["retrieved_ids"],
                retrieved_labels=result["retrieved_labels"],
                similarity_scores=result["similarity_scores"],
                raw_llm_output=result["raw_output"],
            )
            if (i + 1) % 10 == 0:
                click.echo(f"  Classified {i + 1}/{len(texts)}...")

    # Compute and display metrics
    acc = accuracy(predicted_labels, actual_labels)
    mf1 = macro_f1(predicted_labels, actual_labels)
    per_cat = per_category_precision_recall(predicted_labels, actual_labels)

    click.echo(f"\n{'='*60}")
    click.echo(f"Classification Evaluation: {ds['name']} / {strategy}")
    click.echo(f"{'='*60}")
    click.echo(f"Documents:  {len(texts)}")
    click.echo(f"Accuracy:   {acc:.4f}")
    click.echo(f"Macro-F1:   {mf1:.4f}")
    click.echo(f"\nPer-category:")
    click.echo(f"  {'Category':<30} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    click.echo(f"  {'-'*53}")
    for cat in sorted(per_cat.keys()):
        m = per_cat[cat]
        click.echo(f"  {cat:<30} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>5}")

    logger.close()
    click.echo(f"\nResults logged to {cfg['evaluation']['db_path']}")
```

**Design decisions:**
- `classify` handles single-document classification — useful for testing and demo.
- `classify-eval` is the batch evaluation command — runs the full labeled dataset, logs
  every result to SQLite, and prints aggregate metrics.
- `--strategy` can override the config — enables quick comparison:
  `promptrag classify-eval --config configs/covid_poynter.yaml --strategy knn`
  `promptrag classify-eval --config configs/covid_poynter.yaml --strategy rag`
- `--limit` caps the number of documents — essential for testing RAG strategy (LLM inference
  is slow; you don't want to wait for 317 documents during development).
- Progress output every 50 docs (knn) or 10 docs (rag) — the RAG strategy is much slower.

**kNN self-prediction note:** When running kNN evaluation on the same corpus used for ingestion,
the query document will retrieve *itself* as the nearest neighbor (distance ≈ 0). This
inflates accuracy. Two approaches to handle this:
1. **Filter self from results** — add `where` metadata filter in ChromaDB query:
   `collection.query(..., where={"doc_id": {"$ne": doc_id}})`. This is the cleanest approach.
2. **Use k+1 and skip first result** — fragile, only works if self is always rank 1.

Option 1 is recommended. This requires passing `doc_id` to `knn_classify()` and using
ChromaDB's `where` filter. This should be implemented as part of kNN classification.

**Updated `knn_classify` signature for self-exclusion:**
```python
def knn_classify(
    vector_store: VectorStore,
    collection,
    query_text: str,
    k: int = 5,
    weight_by_score: bool = False,
    exclude_doc_id: str | None = None,
) -> dict:
```

When `exclude_doc_id` is provided, the function queries for `k+1` results and filters out
any chunk whose `doc_id` metadata matches. The `classify-eval` command passes the current
document's ID as `exclude_doc_id`.

---

### Step 9: Integration test — COVID/Poynter classification evaluation

**Goal:** End-to-end test of the classification pipeline on the COVID dataset.

**Creates:** `scripts/test_covid_classification.py`

**Sequence:**
```bash
# 1. Re-ingest COVID data with labels in metadata
promptrag ingest --config configs/covid_poynter.yaml

# 2. Run kNN classification evaluation
promptrag classify-eval --config configs/covid_poynter.yaml --strategy knn --k 5

# 3. Run kNN with weighted voting
# (modify config or add --weight flag)

# 4. Run RAG classification on a small subset
promptrag classify-eval --config configs/covid_poynter.yaml --strategy rag --k 5 --limit 20

# 5. Compare results in SQLite
# SELECT strategy, COUNT(*), AVG(predicted_label = actual_label) as accuracy
# FROM classification_runs
# WHERE dataset = 'covid'
# GROUP BY strategy
```

**Expected results:**
- kNN accuracy on COVID should be moderate (0.3–0.6 range) given 19 categories and some
  semantically overlapping categories (e.g., `virus downplaying` vs. `virus trajectory abroad`).
- If kNN accuracy is very high (>0.8), that's either a good sign (embedding model separates
  well) or a leak (self-retrieval not excluded).
- RAG accuracy should be competitive with or better than kNN if the LLM can reason about
  the retrieved examples.
- Categories with very few examples (`celebrities with covid`: 3) will have noisy per-category
  metrics — expected and documented.

**Test point:** The script runs without error, classification_runs table has rows, and
aggregate metrics are printed to stdout.

---

## Implementation Order Summary

```
Step 0: Fix COVID config separator             (5 min, config only)
Step 1: Label plumbing in pipeline.ingest()    (pipeline.py edit, test)
Step 2: Classification metrics in evaluation   (evaluation.py additions, tests)
Step 3: kNN classifier                         (new classifier.py, tests)
Step 4: Anchor-vector classifier               (classifier.py additions, tests)
Step 5: RAG classifier                         (classifier.py additions, tests)
Step 6: Classification logging                 (evaluation.py additions, tests)
Step 7: Config schema extensions               (YAML edits)
Step 8: CLI commands                           (cli.py additions)
Step 9: COVID integration test                 (end-to-end validation)
```

Steps 1–2 are shared plumbing. Steps 3–5 are the three strategies in order of complexity.
Steps 6–8 are infrastructure for running and logging experiments. Step 9 validates everything.

Each step has an isolated test point. No step requires the next step — if a step fails,
the previous steps still work independently.

---

## Scope Exclusions (Explicit)

These items are related but intentionally excluded from Phase 2:

| Item | Why excluded | When to revisit |
|---|---|---|
| Wolf Archive gold set labeling | Requires manual identification of 24 episodes | Phase 3 or dedicated labeling session |
| Cross-encoder reranking before classification | Adds complexity before baseline is established | After Phase 2 baseline metrics exist |
| Constrained decoding for RAG classification | `_parse_label()` is sufficient for now | If LLM parse failures exceed 10% |
| Multi-label classification | All three datasets are single-label | If a multi-label dataset is added |
| Embedding model comparison runs | Phase 1 established the model; this is about strategies | After all three strategies work on one model |
| Prompt engineering for RAG classifier | Default prompt is a starting point | After initial RAG evaluation results |

---

## Decision Points During Implementation

These are questions that will arise during implementation. The answers should be logged in
`docs/decision-log.md` as they are resolved.

1. **kNN self-exclusion:** Does ChromaDB's `where` filter reliably exclude self-matches?
   If not, what's the fallback?
2. **Anchor descriptions for COVID:** The 19 categories don't have natural multi-word
   descriptions like the Wolf Archive themes. How should anchors be constructed?
   (Options: category name only, category name + representative tweet, synthetic description)
3. **RAG parse failure rate:** What percent of LLM outputs fail to match a valid label?
   If >10%, is the prompt the problem or the model?
4. **k sensitivity:** Does kNN accuracy change significantly between k=3, k=5, k=10?
   The answer determines whether k-tuning is Phase 2 or Phase 3 scope.
