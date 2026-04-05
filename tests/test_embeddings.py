import pytest
from promptrag.embeddings import EmbeddingModel


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel(device="cpu")


def test_embedding_dimension(model):
    result = model.encode(["test sentence"], is_query=True)
    assert len(result) == 1
    assert len(result[0]) == model.dimension


def test_batch_encoding(model):
    texts = [f"sentence {i}" for i in range(10)]
    result = model.encode(texts, batch_size=4, is_query=False)
    assert len(result) == 10


def test_device_fallback():
    model = EmbeddingModel(device="cuda")  # not available in test env
    assert model.device == "cpu"
