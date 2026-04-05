import json
import sqlite3
import uuid
from datetime import datetime, timezone
import math


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
