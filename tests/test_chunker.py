import pytest
from promptrag.chunker import chunk_documents


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


def test_multiple_documents():
    texts = ["short doc", " ".join(["word"] * 200)]
    chunks = chunk_documents(texts, ["d1", "d2"], chunk_size=150, chunk_overlap=20)
    assert chunks[0].doc_id == "d1"
    assert chunks[0].chunk_index == 0
    assert any(c.doc_id == "d2" for c in chunks)


def test_metadata_passthrough():
    meta = [{"source": "test"}]
    chunks = chunk_documents(["hello world"], ["d1"], metadata=meta)
    assert chunks[0].metadata == {"source": "test"}
