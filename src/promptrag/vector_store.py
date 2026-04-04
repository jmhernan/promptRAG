import chromadb
from promptrag.embeddings import EmbeddingModel
from promptrag.chunker import Chunk


class VectorStore:
    """ChromaDB-backed vector store with embedding model integration."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_directory: str = "experiments/chromadb",
    ):
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=persist_directory)

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a named collection. Name follows convention:
        {dataset}_{embedding_model_shortname}
        """
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, collection: chromadb.Collection, chunks: list[Chunk]) -> None:
        """Add chunked documents to a collection with metadata."""
        ids = [f"{c.doc_id}_chunk{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"doc_id": c.doc_id, "chunk_index": c.chunk_index, **c.metadata}
            for c in chunks
        ]
        embeddings = self.embedding_model.encode(documents, is_query=False)

        # ChromaDB has a batch limit — add in batches of 5000
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def query(
        self,
        collection: chromadb.Collection,
        query_texts: list[str],
        k: int = 5,
    ) -> dict:
        """Query collection. Returns ChromaDB results dict with
        ids, documents, distances, metadatas.
        """
        query_embeddings = self.embedding_model.encode(query_texts, is_query=True)
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
