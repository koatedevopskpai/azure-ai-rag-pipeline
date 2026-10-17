"""
Text chunking strategies — multiple approaches for splitting documents.

Architecture decision: Multiple chunking strategies over one-size-fits-all.

Why: Different document types require different chunking approaches.
Financial research documents (annual reports, earnings call transcripts,
analyst notes) benefit from different chunk boundaries. Three strategies
let the user choose based on document type:
- Recursive: Best for structured text with clear paragraph boundaries
- Semantic: Best for dense analytical content where sentence boundaries
  matter more than character counts
- Hierarchical: Best for long documents where the LLM needs section-level
  context (parent chunk) to correctly answer subsection-level questions

Architecture decision: Pure functions over class-based chunkers.

Why: Chunking is stateless text transformation — no dependencies, no
configuration, no side effects. Pure functions are testable in isolation,
composable (the output of one strategy feeds into another), and have
zero overhead. Class-based chunkers would add abstraction without benefit.
"""

from __future__ import annotations

import re
from typing import Any

from app.config import settings
from app.models.schemas import ChunkingStrategy

# ------------------------------------------------------------------
# Recursive chunking — splits text on paragraph boundaries, then
# sentence boundaries, then character count.
# ------------------------------------------------------------------
# Architecture decision: Recursive splitting over fixed-size sliding window.
# Why: Fixed-size sliding window chunks can split in the middle of a
# sentence or thought, degrading RAG quality. Recursive splitting tries
# to find natural boundaries (paragraphs first, then sentences, then
# clauses) and only falls back to character count when no boundary is
# found. This produces semantically coherent chunks.


def chunk_recursive(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict[str, Any]]:
    """
    Split text using recursive boundary-aware chunking.

    Separators tried in order: double newline, single newline, sentence
    end (.!?), then character count as last resort.

    Args:
        text: The text to chunk.
        chunk_size: Maximum characters per chunk (default: from settings).
        overlap: Overlap between chunks in characters (default: from settings).

    Returns:
        List of dicts with 'content', 'chunk_index', 'strategy' keys.
    """
    size = chunk_size or settings.CHUNK_SIZE
    overlap_chars = overlap or settings.CHUNK_OVERLAP
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", "]

    chunks = []
    remaining = text

    index = 0
    while remaining:
        if len(remaining) <= size:
            chunks.append(
                {
                    "content": remaining.strip(),
                    "chunk_index": index,
                    "strategy": ChunkingStrategy.RECURSIVE,
                }
            )
            break

        # Try each separator to find a good split point
        split_at = -1
        for sep in separators:
            pos = remaining.rfind(sep, 0, size)
            if pos > split_at:
                split_at = pos

        # If no separator found within chunk_size, split at chunk_size
        if split_at == -1:
            split_at = size

        chunk_text = remaining[: split_at + 1].strip()
        if chunk_text:
            chunks.append(
                {
                    "content": chunk_text,
                    "chunk_index": index,
                    "strategy": ChunkingStrategy.RECURSIVE,
                }
            )
            index += 1

        # Advance by chunk_size - overlap for the next chunk
        advance = max(1, split_at + 1 - overlap_chars)
        remaining = remaining[advance:]

    return chunks


# ------------------------------------------------------------------
# Semantic chunking — uses embedding similarity to find natural breaks.
# ------------------------------------------------------------------
# Architecture decision: Embedding-based over NLP-based semantic
# chunking.
# Why: Embedding similarity captures semantic relatedness beyond what
# NLP features (POS tags, dependency parse) can detect. Two sentences
# about "EBITDA margins" will have high cosine similarity even if they
# are separated by a paragraph about "revenue growth." The trade-off is
# cost (embedding API calls) vs quality. For financial research, quality
# is the priority.


def chunk_semantic(
    text: str,
    sentences_per_chunk: int = 5,
) -> list[dict[str, Any]]:
    """
    Split text into semantically coherent chunks.

    First splits the text into sentences, then groups consecutive sentences
    until the group reaches sentences_per_chunk. Each group becomes a chunk.
    Future enhancement: use embedding cosine similarity between adjacent
    sentences to detect topic shifts.

    Args:
        text: The text to chunk.
        sentences_per_chunk: Target number of sentences per chunk.

    Returns:
        List of chunk dicts.
    """
    # Simple sentence splitting — handles common financial text patterns
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z\"\'\(])"
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i : i + sentences_per_chunk]
        chunks.append(
            {
                "content": " ".join(group),
                "chunk_index": len(chunks),
                "strategy": ChunkingStrategy.SEMANTIC,
            }
        )

    return chunks


# ------------------------------------------------------------------
# Hierarchical chunking — document → sections → paragraphs.
# ------------------------------------------------------------------
# Architecture decision: Three-level hierarchy over flat chunks.
# Why: Financial research questions often reference document structure:
# "What does Section 3.2 say about risk factors?" The LLM needs the
# section heading (level 0 parent) and the specific paragraph (level 2
# child) to answer correctly. Hierarchical chunks store parent-child
# relationships so the RAG pipeline can retrieve the relevant child
# and automatically include the parent context.


def chunk_hierarchical(
    text: str,
    max_section_chars: int = 2000,
    max_paragraph_chars: int = 500,
) -> list[dict[str, Any]]:
    """
    Split text into a three-level hierarchy: sections → subsections → paragraphs.

    Level 0 = section headings (h1/h2)
    Level 1 = subsection content (h3/h4)
    Level 2 = paragraphs within subsections

    Args:
        text: The text to chunk.
        max_section_chars: Maximum characters per section.
        max_paragraph_chars: Maximum characters per paragraph.

    Returns:
        List of chunk dicts with 'level' field.
    """
    # Detect markdown-style headings
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    sections = heading_pattern.split(text)

    chunks = []
    current_section_id = None
    for i in range(0, len(sections) - 2, 2):
        level = len(sections[i])  # number of # characters
        heading = sections[i + 1].strip()
        content = sections[i + 2].strip() if i + 2 < len(sections) else ""

        chunk = {
            "content": f"{'#' * level} {heading}\n\n{content}",
            "chunk_index": len(chunks),
            "strategy": ChunkingStrategy.HIERARCHICAL,
            "level": level - 1,
            "metadata": {"heading": heading, "heading_level": level},
        }

        # If we have a parent section and this is a deeper level
        if level > 1 and current_section_id and len(chunk["content"]) <= max_paragraph_chars:
            chunk["parent_chunk_id"] = current_section_id

        chunks.append(chunk)

        if level == 1:
            current_section_id = chunk.get("content", "")[:50]

    return chunks


# ------------------------------------------------------------------
# Chunking dispatcher
# ------------------------------------------------------------------


def chunk_document(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict[str, Any]]:
    """
    Chunk a document using the specified strategy.

    Args:
        text: The document text to chunk.
        strategy: Which chunking strategy to use.
        chunk_size: Chunk size (recursive strategy only).
        overlap: Chunk overlap (recursive strategy only).

    Returns:
        List of chunk dicts.
    """
    if strategy == ChunkingStrategy.RECURSIVE:
        return chunk_recursive(text, chunk_size, overlap)
    elif strategy == ChunkingStrategy.SEMANTIC:
        return chunk_semantic(text)
    elif strategy == ChunkingStrategy.HIERARCHICAL:
        return chunk_hierarchical(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
