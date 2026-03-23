# Prompt Generator Using RAG

Generate prompts to input to a Large Language Model of your choosing using RAG (Retrieval Augmented Generation). This implementation helps you separate out the prompt engineering process for any LLM task, and helps you scale and validate the prompt generation process.

## Overview

promptRAG takes a knowledge base (e.g. a CSV of labeled misinformation examples), builds a vector index from it, and then for any new input text retrieves the most relevant examples to assemble a context-rich prompt ready to send to an LLM. The current example dataset focuses on COVID misinformation detection using the Poynter fact-check dataset.

## Pipeline

```
Raw Data (CSV)
  │
  ▼  text_preprocess.py
Clean Text
  │
  ▼  create_vdb.py
Generate Embeddings (sentence-transformers)
  │
  ▼
Build FAISS Vector Index
  │
  ▼  prompt_generator.py
Query Index for Similar Documents
  │
  ▼
Format Prompt from Template
  │
  ▼
LLM-Ready Prompt
```

## Project Structure

```
src/
  text_preprocess.py      # Text cleaning utilities
  create_vdb.py           # Embedding generation & FAISS index creation
  prompt_generator.py     # RAG prompt assembly
  transformer_playground.py  # Educational transformer implementation (not part of main pipeline)
  prompt_templates/
    promptTemplate.txt                    # Generic prompt template
    promptTemplate_mistral_onthefly.txt   # Mistral-specific prompt template
data/
  poynter_coded_breon_tab.csv   # Poynter COVID misinformation dataset
example_script.py       # End-to-end CLI demo
example_notebook.ipynb  # Interactive Jupyter notebook demo
environment.yml         # Conda environment specification
```

## Steps

### 1. Text Preprocessing (`src/text_preprocess.py`)

Cleans raw text from the knowledge base:

- Removes symbols and non-alphanumeric characters
- Normalizes whitespace and converts to lowercase
- Standardizes COVID-related terms (e.g. → `covid_19`)
- Strips non-ASCII characters and empty entries
- Supports parallel processing for large datasets via `ProcessPoolExecutor`

Also provides helper utilities like `get_top_n_words()` for corpus frequency analysis.

### 2. Embedding & Indexing (`src/create_vdb.py`)

Generates vector embeddings and builds a FAISS index for fast similarity search:

- Loads the `multi-qa-mpnet-base-dot-v1` sentence-transformer model (auto-detects GPU/CPU)
- Tokenizes text and produces 768-dimensional embeddings using CLS pooling
- Wraps the data in a HuggingFace `Dataset` and adds an embeddings column
- Creates a FAISS index on the dataset for efficient nearest-neighbor lookup

### 3. Prompt Generation (`src/prompt_generator.py`)

The core RAG step — retrieves relevant examples and assembles the final prompt:

- Queries the FAISS index to find the top-k most similar documents to a given input
- Formats retrieved documents as bullet points with similarity scores
- Fills a prompt template with the input text and retrieved context
- Returns the complete prompt along with the retrieved texts, discussion themes, and similarity scores

### 4. Prompt Templates (`src/prompt_templates/`)

Two templates are included:

- **`promptTemplate.txt`** — Generic template with placeholders for retrieved examples and the target statement
- **`promptTemplate_mistral_onthefly.txt`** — Mistral-formatted template using `[INST]` tokens, with an explicit classification instruction

Templates use `{retrieved_texts}` and `{tweet}` placeholders that get filled at generation time.

## Usage

### Script

```bash
python example_script.py
```

Loads the Poynter dataset, builds the vector index, and generates prompts for a set of sample tweets.

### Notebook

Open `example_notebook.ipynb` for an interactive walkthrough of the same pipeline.

## Setup

```bash
conda env create -f environment.yml
conda activate twitter_pytorch
```

### Key Dependencies

- **Embeddings**: `sentence-transformers`, `transformers`, HuggingFace `datasets`
- **Vector Search**: `faiss-cpu`
- **Deep Learning**: `pytorch`
- **Data**: `pandas`, `numpy`
- **NLP**: `nltk`, `gensim`, `scikit-learn`

## Additional: Transformer Playground

`src/transformer_playground.py` is a standalone educational implementation of transformer components (multi-head attention, feed-forward layers, encoder stack) with attention visualization. It is **not** part of the main RAG pipeline.
