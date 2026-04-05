from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_documents(
    texts: list[str],
    doc_ids: list[str],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    metadata: list[dict] | None = None,
) -> list[Chunk]:
    """Split documents into fixed-size chunks by word count.

    Documents shorter than chunk_size are passed through as a single chunk.
    """
    chunks = []
    for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
        doc_meta = metadata[i] if metadata else {}
        words = text.split()

        if len(words) <= chunk_size:
            chunks.append(Chunk(
                text=text,
                doc_id=doc_id,
                chunk_index=0,
                metadata=doc_meta,
            ))
            continue

        start = 0
        chunk_idx = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=chunk_idx,
                metadata=doc_meta,
            ))
            start += chunk_size - chunk_overlap
            chunk_idx += 1

    return chunks
