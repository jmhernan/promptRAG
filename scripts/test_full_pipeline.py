"""Full pipeline test — requires LLM download (~7.6GB first run).

Ingests Wolf Archive, runs a query through retrieval + generation,
and logs the run to SQLite.
"""
from promptrag.pipeline import RAGPipeline, load_config


def main():
    config = load_config("../configs/wolf_archive.yaml")
    pipeline = RAGPipeline(config)

    # Ingest if collection is empty
    if pipeline.collection.count() == 0:
        print("Ingesting Wolf Archive...")
        count = pipeline.ingest()
        print(f"Ingested {count} chunks")
    else:
        print(f"Collection already has {pipeline.collection.count()} documents")

    # Run queries
    queries = [
        "children playing a pretend war game",
        "children fighting or in physical conflict",
        "cooperative play between children",
    ]

    for query_text in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query_text}")
        print(f"{'='*80}")

        result = pipeline.query(query_text)

        print(f"\nRun ID: {result['run_id']}")
        print(f"\nRetrieved {len(result['retrieved_docs'])} documents:")
        for doc_id, score in zip(result["retrieved_ids"], result["scores"]):
            print(f"  [{doc_id}] score: {score:.4f}")

        print(f"\nLLM Output length: {len(result['llm_output'])} chars (logged to SQLite)")

    pipeline.close()
    print("\nDone. Results logged to experiments/runs.db")


if __name__ == "__main__":
    main()
