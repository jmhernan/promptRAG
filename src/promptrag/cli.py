import click
from promptrag.pipeline import RAGPipeline, load_config


@click.group()
def main():
    """promptrag — Research-team-grade RAG pipeline."""
    pass


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
def ingest(config):
    """Ingest dataset into vector store."""
    cfg = load_config(config)
    pipeline = RAGPipeline(cfg)
    count = pipeline.ingest()
    click.echo(f"Ingested {count} chunks into {cfg['vector_store']['collection_name']}")
    pipeline.close()


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
@click.option("--k", default=None, type=int, help="Number of documents to retrieve")
@click.option("--verbose", is_flag=True, help="Show retrieved text and LLM output (may contain sensitive data)")
@click.argument("query_text")
def query(config, k, verbose, query_text):
    """Run a query through the pipeline."""
    cfg = load_config(config)
    pipeline = RAGPipeline(cfg)
    result = pipeline.query(query_text, k=k)
    click.echo(f"\nRun ID: {result['run_id']}")
    click.echo(f"\nRetrieved {len(result['retrieved_docs'])} documents:")
    for doc_id, score in zip(result["retrieved_ids"], result["scores"]):
        click.echo(f"  [{doc_id}] (score: {score:.4f})")
    if verbose:
        for doc_id, text in zip(result["retrieved_ids"], result["retrieved_docs"]):
            click.echo(f"\n  [{doc_id}] {text[:200]}...")
        click.echo(f"\nLLM Output:\n{result['llm_output']}")
    else:
        click.echo(f"\nLLM Output length: {len(result['llm_output'])} chars (use --verbose to display)")
    pipeline.close()
