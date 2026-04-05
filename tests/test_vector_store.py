import pytest
from promptrag.embeddings import EmbeddingModel
from promptrag.vector_store import VectorStore
from promptrag.chunker import Chunk


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel(device="cpu")


def test_add_and_query(tmp_path, model):
    store = VectorStore(model, persist_directory=str(tmp_path / "chroma"))
    collection = store.get_or_create_collection("test_collection")

    chunks = [
        Chunk(text="the cat sat on the mat", doc_id="d1", chunk_index=0),
        Chunk(text="dogs love to play fetch", doc_id="d2", chunk_index=0),
    ]
    store.add_chunks(collection, chunks)

    results = store.query(collection, ["feline sitting"], k=2)
    # Verify structure — don't assert semantic ranking with base (non-fine-tuned) model
    assert len(results["ids"][0]) == 2
    assert set(results["ids"][0]) == {"d1_chunk0", "d2_chunk0"}
    assert len(results["documents"][0]) == 2
    assert len(results["distances"][0]) == 2


def test_collection_persistence(tmp_path, model):
    persist_dir = str(tmp_path / "chroma")

    store1 = VectorStore(model, persist_directory=persist_dir)
    coll1 = store1.get_or_create_collection("persist_test")
    chunks = [Chunk(text="persistence test", doc_id="d1", chunk_index=0)]
    store1.add_chunks(coll1, chunks)

    # New client instance, same directory — data should persist
    store2 = VectorStore(model, persist_directory=persist_dir)
    coll2 = store2.get_or_create_collection("persist_test")
    assert coll2.count() == 1
