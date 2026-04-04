"""Test retrieval only — no LLM download needed.

Ingests Wolf Archive into ChromaDB, runs sample queries,
and prints top-k results for manual inspection.
"""
from promptrag.embeddings import EmbeddingModel
from promptrag.vector_store import VectorStore
from promptrag.chunker import chunk_documents
import polars as pl


def main():
    print("Loading embedding model...")
    model = EmbeddingModel(device="mps")

    store = VectorStore(model)
    collection_name = "wolf_archive_modernbert_embed"
    coll = store.get_or_create_collection(collection_name)

    # Only ingest if collection is empty
    if coll.count() == 0:
        print("Ingesting Wolf Archive...")
        df = pl.read_csv("../data/4ChildObservation_MasterFile.csv")
        df = df.filter(pl.col("text").is_not_null())
        texts = df["text"].to_list()
        ids = [str(x) for x in df["Index"].to_list()]
        chunks = chunk_documents(texts, ids)
        store.add_chunks(coll, chunks)
        print(f"Ingested {len(chunks)} chunks into {collection_name}")
    else:
        print(f"Collection {collection_name} already has {coll.count()} documents")

    # Sample queries — mix of themes from the Wolf Archive
    queries = [
        "children playing a pretend war game",
        "children fighting or in physical conflict",
        "cooperative play between children",
        "children shopping at a store",
        "children at school",
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        results = store.query(coll, [query], k=5)
        for doc_id, dist, text in zip(
            results["ids"][0], results["distances"][0], results["documents"][0]
        ):
            score = 1.0 - dist
            print(f"  [{doc_id}] (score: {score:.4f})")


if __name__ == "__main__":
    main()
