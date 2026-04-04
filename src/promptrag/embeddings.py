import torch
torch.backends.mps.enable_fallback_for_missing_ops = True

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wraps SentenceTransformer with device detection and batch encoding.

    Some models (e.g. nomic-ai/modernbert-embed-base) require input prefixes:
    - 'search_query: ' for queries
    - 'search_document: ' for documents
    Set query_prefix/document_prefix accordingly.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/modernbert-embed-base",
        device: str = "mps",
        query_prefix: str = "search_query: ",
        document_prefix: str = "search_document: ",
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def encode(self, texts: list[str], batch_size: int = 32, is_query: bool = False) -> list[list[float]]:
        """Encode texts using SentenceTransformer.encode() — handles pooling correctly.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            is_query: If True, prepend query_prefix; otherwise prepend document_prefix.
        """
        prefix = self.query_prefix if is_query else self.document_prefix
        prefixed_texts = [f"{prefix}{t}" for t in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
