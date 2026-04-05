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
