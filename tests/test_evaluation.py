import pytest
import sqlite3
from promptrag.evaluation import precision_at_k, recall_at_k, mrr, ndcg_at_k, RunLogger


def test_precision_at_k():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d1", "d3", "d5"}
    assert precision_at_k(retrieved, relevant, 5) == 0.6
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d3", "d5"}
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


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
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row is not None
    conn.close()
    logger.close()
