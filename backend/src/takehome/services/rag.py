from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import structlog
import tiktoken
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.config import settings  # noqa: F401 — triggers env export
from takehome.db.models import Document, DocumentChunk

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Shared clients
# ---------------------------------------------------------------------------

_openai: AsyncOpenAI | None = None
_encoder: tiktoken.Encoding | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI()
    return _openai


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# Page boundary marker produced by document.py during extraction
PAGE_MARKER_RE = re.compile(r"^--- Page (\d+) ---$", re.MULTILINE)


@dataclass
class ChunkInfo:
    content: str
    page_number: int
    section_header: str | None
    token_count: int


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    doc_label: str
    doc_filename: str
    content: str
    context_text: str | None
    page_number: int
    section_header: str | None
    score: float


# ---------------------------------------------------------------------------
# 1. Chunking
# ---------------------------------------------------------------------------


def chunk_document(extracted_text: str) -> list[ChunkInfo]:
    """Split extracted text into chunks at page boundaries, targeting ~500 tokens."""
    if not extracted_text.strip():
        return []

    # Split by page markers
    pages: list[tuple[int, str]] = []
    parts = PAGE_MARKER_RE.split(extracted_text)

    # parts alternates: [pre-text, page_num, page_text, page_num, page_text, ...]
    if parts[0].strip():
        pages.append((1, parts[0].strip()))
    for i in range(1, len(parts) - 1, 2):
        page_num = int(parts[i])
        page_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if page_text:
            pages.append((page_num, page_text))

    if not pages:
        # Fallback: treat entire text as page 1
        pages = [(1, extracted_text.strip())]

    chunks: list[ChunkInfo] = []
    encoder = _get_encoder()

    for page_num, page_text in pages:
        paragraphs = page_text.split("\n\n")

        current_chunk_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = len(encoder.encode(para))

            # If adding this paragraph would exceed target, flush current chunk
            if current_tokens > 0 and current_tokens + para_tokens > CHUNK_TARGET_TOKENS:
                chunk_text = "\n\n".join(current_chunk_parts)
                chunks.append(
                    ChunkInfo(
                        content=chunk_text,
                        page_number=page_num,
                        section_header=None,
                        token_count=current_tokens,
                    )
                )
                # Overlap: keep the last part if it's small enough
                if current_chunk_parts and _count_tokens(current_chunk_parts[-1]) <= CHUNK_OVERLAP_TOKENS:
                    overlap_part = current_chunk_parts[-1]
                    current_chunk_parts = [overlap_part]
                    current_tokens = _count_tokens(overlap_part)
                else:
                    current_chunk_parts = []
                    current_tokens = 0

            current_chunk_parts.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_chunk_parts:
            chunk_text = "\n\n".join(current_chunk_parts)
            chunks.append(
                ChunkInfo(
                    content=chunk_text,
                    page_number=page_num,
                    section_header=None,
                    token_count=current_tokens,
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# 2. Contextual Retrieval — generate context per chunk
# ---------------------------------------------------------------------------


@dataclass
class ChunkMetadata:
    context: str
    section: str | None


async def generate_chunk_contexts(
    document_text: str, chunks: list[ChunkInfo]
) -> list[ChunkMetadata]:
    """Use Claude Haiku to generate context and identify section/clause for each chunk.

    Returns a list of ChunkMetadata (context string + section identifier), one per chunk.
    """
    import json as _json

    import anthropic

    client = anthropic.AsyncAnthropic()

    # Truncate document text if very long (Haiku context is 200k but be reasonable)
    max_doc_chars = 100_000
    doc_summary = document_text[:max_doc_chars]
    if len(document_text) > max_doc_chars:
        doc_summary += "\n\n[... document truncated for context generation ...]"

    results: list[ChunkMetadata] = []

    for chunk in chunks:
        prompt = (
            "<document>\n"
            f"{doc_summary}\n"
            "</document>\n\n"
            "Here is the chunk we want to situate within the whole document:\n"
            "<chunk>\n"
            f"{chunk.content}\n"
            "</chunk>\n\n"
            "Return a JSON object with exactly two fields:\n"
            "1. \"context\": A short succinct context (2-3 sentences) to situate this chunk "
            "within the overall document for the purposes of improving search retrieval. "
            "If this is a legal document, mention the document type, relevant section/clause, "
            "parties involved, and any defined terms.\n"
            "2. \"section\": The specific section, clause, or article identifier this chunk falls under "
            "(e.g. \"Section 3 — Rent\", \"4.1 Tenant's Obligations\", \"Clause 7.2\", "
            "\"Executive Summary\"). Use the exact heading from the document. "
            "If no clear section applies, use null.\n\n"
            "Return ONLY the JSON object, no other text."
        )

        try:
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()  # type: ignore[union-attr]

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            # Parse JSON response
            try:
                parsed = _json.loads(raw)
                context = parsed.get("context", "")
                section = parsed.get("section")
                # Normalize null/empty section
                if not section or section == "null":
                    section = None
                results.append(ChunkMetadata(context=context, section=section))
            except _json.JSONDecodeError:
                # Fallback: treat entire response as context
                logger.warning("Failed to parse JSON from Haiku, using raw text", page=chunk.page_number)
                results.append(ChunkMetadata(context=raw, section=None))
        except Exception:
            logger.exception("Failed to generate context for chunk", page=chunk.page_number)
            results.append(ChunkMetadata(context="", section=None))

    return results


# ---------------------------------------------------------------------------
# 3. Embedding
# ---------------------------------------------------------------------------


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI text-embedding-3-small."""
    if not texts:
        return []

    client = _get_openai()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# 4. Full ingestion pipeline
# ---------------------------------------------------------------------------


async def process_document(session: AsyncSession, document: Document) -> list[DocumentChunk]:
    """Full ingestion: chunk -> contextualize -> embed -> store.

    Called after document upload.
    """
    if not document.extracted_text:
        logger.warning("No extracted text for document", document_id=document.id)
        return []

    logger.info("Starting RAG processing", document_id=document.id, filename=document.filename)

    # 1. Chunk
    chunks = chunk_document(document.extracted_text)
    if not chunks:
        logger.warning("No chunks produced", document_id=document.id)
        return []
    logger.info("Chunked document", document_id=document.id, num_chunks=len(chunks))

    # 2. Contextual retrieval — generate context + section per chunk (via Haiku)
    metadata = await generate_chunk_contexts(document.extracted_text, chunks)

    # 3. Prepare contextualized texts for embedding
    texts_to_embed = []
    for chunk, meta in zip(chunks, metadata):
        if meta.context:
            texts_to_embed.append(f"{meta.context}\n\n{chunk.content}")
        else:
            texts_to_embed.append(chunk.content)

    # 4. Embed (batch)
    embeddings = await embed_texts(texts_to_embed)
    logger.info("Embedded chunks", document_id=document.id, num_embeddings=len(embeddings))

    # 5. Store
    db_chunks: list[DocumentChunk] = []
    for i, (chunk, meta, emb) in enumerate(zip(chunks, metadata, embeddings)):
        db_chunk = DocumentChunk(
            id=uuid.uuid4().hex[:16],
            document_id=document.id,
            chunk_index=i,
            content=chunk.content,
            context_text=meta.context if meta.context else None,
            page_number=chunk.page_number,
            section_header=meta.section,
            embedding=emb,
            token_count=chunk.token_count,
        )
        session.add(db_chunk)
        db_chunks.append(db_chunk)

    await session.commit()
    logger.info("Stored chunks in DB", document_id=document.id, num_chunks=len(db_chunks))

    return db_chunks


# ---------------------------------------------------------------------------
# 5. Hybrid search: pgvector + BM25 + Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


async def search_chunks(
    query: str,
    conversation_id: str,
    session: AsyncSession,
    top_k: int = 10,
) -> list[SearchResult]:
    """Hybrid search: vector similarity + BM25 keyword search merged with RRF."""
    # Embed the query
    query_embeddings = await embed_texts([query])
    if not query_embeddings:
        return []
    query_embedding = query_embeddings[0]

    # Load all chunks for this conversation's documents
    stmt = (
        select(DocumentChunk, Document.label, Document.filename)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.conversation_id == conversation_id)
    )
    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return []

    # -- Vector search (pgvector cosine similarity) --
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    vector_sql = text(
        """
        SELECT dc.id, (dc.embedding <=> CAST(:query_vec AS vector)) AS distance
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.conversation_id = :conv_id
          AND dc.embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT :limit
        """
    )
    vector_result = await session.execute(
        vector_sql,
        {"query_vec": embedding_str, "conv_id": conversation_id, "limit": 20},
    )
    vector_rows = vector_result.all()
    vector_ranks: dict[str, int] = {row[0]: rank for rank, row in enumerate(vector_rows)}

    # -- BM25 keyword search --
    chunk_map: dict[str, tuple] = {}  # chunk_id -> (chunk, label, filename)
    corpus: list[list[str]] = []
    chunk_ids: list[str] = []

    for row in rows:
        chunk = row[0]
        label = row[1]
        filename = row[2]
        chunk_map[chunk.id] = (chunk, label, filename)
        # Tokenize for BM25
        combined = chunk.content
        if chunk.context_text:
            combined = chunk.context_text + " " + combined
        corpus.append(combined.lower().split())
        chunk_ids.append(chunk.id)

    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(
        enumerate(bm25_scores), key=lambda x: x[1], reverse=True
    )[:20]
    bm25_ranks: dict[str, int] = {
        chunk_ids[idx]: rank for rank, (idx, _score) in enumerate(bm25_ranked)
    }

    # -- Reciprocal Rank Fusion --
    k = 60  # RRF constant
    all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores: dict[str, float] = {}
    for cid in all_chunk_ids:
        score = 0.0
        if cid in vector_ranks:
            score += 1.0 / (k + vector_ranks[cid])
        if cid in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[cid])
        rrf_scores[cid] = score

    # Sort by RRF score descending
    top_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    # Build results
    results: list[SearchResult] = []
    for cid in top_ids:
        if cid not in chunk_map:
            continue
        chunk, label, filename = chunk_map[cid]
        results.append(
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                doc_label=label or "Doc",
                doc_filename=filename,
                content=chunk.content,
                context_text=chunk.context_text,
                page_number=chunk.page_number,
                section_header=chunk.section_header,
                score=rrf_scores[cid],
            )
        )

    return results


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results for Claude to consume."""
    if not results:
        return "No relevant passages found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        header = f"[Result {i}] {r.doc_label} ({r.doc_filename}), Page {r.page_number}"
        if r.section_header:
            header += f", Section: {r.section_header}"
        parts.append(f"{header}\n{r.content}")

    return "\n\n---\n\n".join(parts)
