"""
End-to-end RAG evaluation suite across ALL synthetic documents.

Evaluates:
  1. Ingestion quality    — chunking coverage, context generation
  2. Retrieval quality    — precision, cross-doc coverage, MRR
  3. Agentic tool use     — does Sonnet call search? how many times? query quality?
  4. Citation grounding   — are citations real? do they reference actual pages/docs?
  5. Response quality     — does the answer address the question?

Uses Anthropic API directly for full tool-call transparency.

Usage:
    uv run python test_rag_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend", "src"))

from dotenv import load_dotenv

load_dotenv()

import anthropic
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi

from takehome.services.rag import (
    ChunkInfo,
    SearchResult,
    chunk_document,
    embed_texts,
    format_search_results,
    generate_chunk_contexts,
)
from takehome.services.llm import SYSTEM_PROMPT, extract_citations, strip_cite_tags


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        if text.strip():
            pages.append(f"--- Page {i + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def hdr(title: str) -> None:
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def step(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 54 - len(title))}")


# ─────────────────────────────────────────────────────
# In-memory corpus (replaces pgvector)
# ─────────────────────────────────────────────────────

class InMemoryCorpus:
    def __init__(self) -> None:
        self.chunks: list[ChunkInfo] = []
        self.contexts: list[str] = []
        self.embeddings: list[list[float]] = []
        self.doc_labels: list[str] = []
        self.doc_filenames: list[str] = []

    def add(self, chunks: list[ChunkInfo], contexts: list[str],
            embeddings: list[list[float]], label: str, filename: str) -> None:
        for c, ctx, emb in zip(chunks, contexts, embeddings):
            self.chunks.append(c)
            self.contexts.append(ctx)
            self.embeddings.append(emb)
            self.doc_labels.append(label)
            self.doc_filenames.append(filename)

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_emb = (await embed_texts([query]))[0]

        # Vector ranking
        sims = [(i, cosine_sim(query_emb, e)) for i, e in enumerate(self.embeddings)]
        sims.sort(key=lambda x: x[1], reverse=True)
        vector_ranks = {i: rank for rank, (i, _) in enumerate(sims[:20])}

        # BM25 ranking
        corpus = []
        for c, ctx in zip(self.chunks, self.contexts):
            corpus.append(((ctx + " " + c.content) if ctx else c.content).lower().split())
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        bm25_sorted = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:20]
        bm25_ranks = {i: rank for rank, (i, _) in enumerate(bm25_sorted)}

        # RRF
        k = 60
        all_ids = set(vector_ranks) | set(bm25_ranks)
        rrf: dict[int, float] = {}
        for idx in all_ids:
            s = 0.0
            if idx in vector_ranks:
                s += 1.0 / (k + vector_ranks[idx])
            if idx in bm25_ranks:
                s += 1.0 / (k + bm25_ranks[idx])
            rrf[idx] = s

        top = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]
        results = []
        for idx in top:
            c = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=f"chunk_{idx}",
                document_id=f"doc_{self.doc_labels[idx]}",
                doc_label=self.doc_labels[idx],
                doc_filename=self.doc_filenames[idx],
                content=c.content,
                context_text=self.contexts[idx],
                page_number=c.page_number,
                section_header=c.section_header,
                score=rrf[idx],
            ))
        return results

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def doc_label_set(self) -> set[str]:
        return set(self.doc_labels)

    def pages_for_doc(self, label: str) -> set[int]:
        return {c.page_number for c, l in zip(self.chunks, self.doc_labels) if l == label}


# ─────────────────────────────────────────────────────
# Agentic chat loop using Anthropic API directly
# ─────────────────────────────────────────────────────

TOOL_DEF = {
    "name": "search_documents",
    "description": (
        "Search uploaded documents for relevant passages. "
        "Call multiple times with different queries for comprehensive results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — be specific about what you're looking for.",
            },
        },
        "required": ["query"],
    },
}


async def run_agentic_chat(
    question: str,
    corpus: InMemoryCorpus,
    client: anthropic.AsyncAnthropic,
) -> dict:
    """Run a full agentic loop: Sonnet + tool calls + final response.

    Returns a dict with the full trace for eval scoring.
    """
    messages: list[dict] = [{"role": "user", "content": question}]
    tool_calls: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0

    max_turns = 5
    for turn in range(max_turns):
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[TOOL_DEF],
            messages=messages,
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Check if the model wants to use a tool
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_use_blocks:
            # Final answer — no more tool calls
            final_text = "\n".join(b.text for b in text_blocks)
            return {
                "response": final_text,
                "tool_calls": tool_calls,
                "turns": turn + 1,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            }

        # Process tool calls
        tool_results = []
        for block in tool_use_blocks:
            query = block.input.get("query", "")
            results = await corpus.search(query, top_k=10)
            formatted = format_search_results(results)

            tool_calls.append({
                "query": query,
                "num_results": len(results),
                "docs_hit": sorted(set(r.doc_label for r in results)),
                "top_3": [f"{r.doc_label} p.{r.page_number}" for r in results[:3]],
                "result_preview": formatted[:200],
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": formatted,
            })

        # Add assistant message + tool results for next turn
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # Exceeded max turns — return what we have
    return {
        "response": "(max turns exceeded)",
        "tool_calls": tool_calls,
        "turns": max_turns,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


# ─────────────────────────────────────────────────────
# Eval scoring
# ─────────────────────────────────────────────────────

def score_retrieval(results: list[SearchResult], expected_docs: list[str]) -> dict:
    """Score retrieval quality for a single query."""
    docs_retrieved = set(r.doc_label for r in results)
    expected = set(expected_docs)
    recall = len(docs_retrieved & expected) / len(expected) if expected else 1.0

    # MRR for first expected doc
    mrr = 0.0
    for i, r in enumerate(results):
        if r.doc_label in expected:
            mrr = 1.0 / (i + 1)
            break

    return {"recall": recall, "mrr": mrr, "docs_retrieved": sorted(docs_retrieved)}


def score_citations(response: str, corpus: InMemoryCorpus) -> dict:
    """Score citation quality."""
    citations = extract_citations(response)
    if not citations:
        return {"count": 0, "grounded": 0, "grounding_rate": 0.0, "details": []}

    valid_docs = corpus.doc_label_set
    valid_pages = {label: corpus.pages_for_doc(label) for label in valid_docs}

    grounded = 0
    details = []
    for c in citations:
        doc_valid = c.doc_label in valid_docs
        page_valid = c.page in valid_pages.get(c.doc_label, set()) if doc_valid else False
        is_grounded = doc_valid and page_valid
        if is_grounded:
            grounded += 1
        details.append({
            "doc": c.doc_label,
            "page": c.page,
            "doc_valid": doc_valid,
            "page_valid": page_valid,
            "grounded": is_grounded,
            "text": c.text[:50],
        })

    return {
        "count": len(citations),
        "grounded": grounded,
        "grounding_rate": grounded / len(citations),
        "details": details,
    }


def score_tool_use(tool_calls: list[dict], expected_docs: list[str]) -> dict:
    """Score how effectively the agent used the search tool."""
    if not tool_calls:
        return {"used_tool": False, "num_calls": 0, "score": 0.0}

    # Check if expected docs were found across all tool calls
    all_docs_hit: set[str] = set()
    for tc in tool_calls:
        all_docs_hit.update(tc["docs_hit"])

    expected = set(expected_docs)
    coverage = len(all_docs_hit & expected) / len(expected) if expected else 1.0

    # Penalize excessive calls (>3 is overkill for these questions)
    efficiency = min(1.0, 3.0 / max(len(tool_calls), 1))

    score = (coverage * 0.7) + (efficiency * 0.3)

    return {
        "used_tool": True,
        "num_calls": len(tool_calls),
        "docs_covered": sorted(all_docs_hit),
        "coverage": coverage,
        "efficiency": efficiency,
        "score": score,
    }


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

async def main() -> None:
    docs_dir = os.path.join(os.path.dirname(__file__), "synthetic-docs")
    pdf_files = sorted(f for f in os.listdir(docs_dir) if f.endswith(".pdf"))
    labels = ["Doc A", "Doc B", "Doc C"]

    hdr("RAG Evaluation Suite — All Synthetic Documents")
    print(f"Documents: {len(pdf_files)}")
    for f, lbl in zip(pdf_files, labels):
        print(f"  [{lbl}] {f}")

    corpus = InMemoryCorpus()

    # ═══ Phase 1: Ingest ═══════════════════════════════
    hdr("Phase 1 — Document Ingestion")

    ingestion_scores: list[dict] = []

    for filename, label in zip(pdf_files, labels):
        step(f"{label}: {filename}")
        path = os.path.join(docs_dir, filename)
        text = extract_text(path)
        print(f"  Extracted: {len(text):,} chars")

        t0 = time.time()
        chunks = chunk_document(text)
        chunk_time = time.time() - t0
        print(f"  Chunks: {len(chunks)} ({chunk_time:.2f}s)")
        for i, c in enumerate(chunks):
            sec = c.section_header or ""
            preview = c.content[:65].replace("\n", " ")
            print(f"    [{i}] p.{c.page_number:>2} {c.token_count:>4}tok  {sec:25s} \"{preview}\"")

        t0 = time.time()
        contexts = await generate_chunk_contexts(text, chunks)
        ctx_time = time.time() - t0
        non_empty = sum(1 for c in contexts if c.strip())
        print(f"  Contexts: {non_empty}/{len(contexts)} non-empty ({ctx_time:.1f}s)")
        for i, ctx in enumerate(contexts):
            short = ctx[:90].replace("\n", " ")
            print(f"    [{i}] \"{short}...\"")

        texts_to_embed = [
            f"{ctx}\n\n{c.content}" if ctx else c.content
            for c, ctx in zip(chunks, contexts)
        ]
        t0 = time.time()
        embeddings = await embed_texts(texts_to_embed)
        emb_time = time.time() - t0
        print(f"  Embeddings: {len(embeddings)} x {len(embeddings[0])}d ({emb_time:.2f}s)")

        corpus.add(chunks, contexts, embeddings, label, filename)

        ingestion_scores.append({
            "doc": label,
            "chunks": len(chunks),
            "context_rate": non_empty / len(chunks) if chunks else 0,
            "avg_tokens": sum(c.token_count for c in chunks) / len(chunks) if chunks else 0,
        })

    print(f"\nTotal corpus: {corpus.size} chunks across {len(pdf_files)} documents")

    # ═══ Phase 2: Retrieval eval ══════════════════════
    hdr("Phase 2 — Retrieval Quality")

    retrieval_tests = [
        {
            "query": "tenant obligations and covenants in the lease",
            "expected_docs": ["Doc A"],
        },
        {
            "query": "environmental contamination and soil pollution risks",
            "expected_docs": ["Doc B"],
        },
        {
            "query": "property title ownership and registered charges",
            "expected_docs": ["Doc C"],
        },
        {
            "query": "insurance requirements and indemnity provisions",
            "expected_docs": ["Doc A"],
        },
        {
            "query": "environmental risks related to the leased property",
            "expected_docs": ["Doc A", "Doc B"],
        },
    ]

    retrieval_scores: list[dict] = []
    for test in retrieval_tests:
        step(f"Query: \"{test['query']}\"")
        t0 = time.time()
        results = await corpus.search(test["query"], top_k=5)
        elapsed = time.time() - t0

        scores = score_retrieval(results, test["expected_docs"])
        retrieval_scores.append(scores)

        print(f"  Results ({elapsed:.3f}s):")
        for i, r in enumerate(results):
            preview = r.content[:70].replace("\n", " ")
            print(f"    #{i+1} [{r.doc_label}] p.{r.page_number} rrf={r.score:.6f} \"{preview}\"")
        print(f"  Expected: {test['expected_docs']} → Retrieved: {scores['docs_retrieved']}")
        print(f"  Recall: {scores['recall']:.2f}  MRR: {scores['mrr']:.2f}")

    # ═══ Phase 3: Agentic chat eval ══════════════════
    hdr("Phase 3 — Agentic Chat (Sonnet + Tool Use)")

    client = anthropic.AsyncAnthropic()

    eval_questions = [
        {
            "question": "What are the key obligations of the tenant in the commercial lease?",
            "expected_docs": ["Doc A"],
            "description": "Single-doc: tenant obligations",
        },
        {
            "question": "What environmental risks were identified in the site assessment, and are any of them addressed in the lease agreement?",
            "expected_docs": ["Doc A", "Doc B"],
            "description": "Cross-doc: env risks vs lease",
        },
        {
            "question": "What is the break clause period and what conditions must the tenant satisfy to exercise it?",
            "expected_docs": ["Doc A"],
            "description": "Specific clause lookup",
        },
        {
            "question": "Summarize any title defects or charges noted in the title report and explain whether they affect the lease.",
            "expected_docs": ["Doc A", "Doc C"],
            "description": "Cross-doc: title + lease",
        },
    ]

    chat_scores: list[dict] = []

    for eq in eval_questions:
        step(f"{eq['description']}")
        print(f"  Q: \"{eq['question']}\"")
        print(f"  Expected docs: {eq['expected_docs']}")

        t0 = time.time()
        trace = await run_agentic_chat(eq["question"], corpus, client)
        elapsed = time.time() - t0

        response = trace["response"]
        citations = extract_citations(response)
        cit_scores = score_citations(response, corpus)
        tool_scores = score_tool_use(trace["tool_calls"], eq["expected_docs"])

        print(f"\n  Time: {elapsed:.1f}s | Turns: {trace['turns']} | Tokens: {trace['input_tokens']}in/{trace['output_tokens']}out")

        # Tool calls
        print(f"\n  Tool calls ({len(trace['tool_calls'])}):")
        for i, tc in enumerate(trace["tool_calls"]):
            print(f"    [{i+1}] query=\"{tc['query']}\"")
            print(f"         → {tc['num_results']} results, docs: {tc['docs_hit']}, top: {tc['top_3']}")

        # Citations
        print(f"\n  Citations ({cit_scores['count']}):")
        for d in cit_scores["details"]:
            status = "GROUNDED" if d["grounded"] else "UNGROUNDED"
            print(f"    [{status}] {d['doc']} p.{d['page']} \"{d['text']}\"")
        print(f"  Grounding rate: {cit_scores['grounding_rate']:.0%} ({cit_scores['grounded']}/{cit_scores['count']})")

        # Response preview
        stripped = strip_cite_tags(response)
        preview = stripped[:250].replace("\n", " ")
        print(f"\n  Response: \"{preview}...\"")

        # Scores
        print(f"\n  Scores:")
        print(f"    Tool use score:     {tool_scores['score']:.2f}  (coverage={tool_scores['coverage']:.2f}, efficiency={tool_scores['efficiency']:.2f})")
        print(f"    Citation grounding: {cit_scores['grounding_rate']:.2f}")
        print(f"    Has citations:      {'Yes' if cit_scores['count'] > 0 else 'No'}")

        chat_scores.append({
            "description": eq["description"],
            "tool_use": tool_scores["score"],
            "grounding": cit_scores["grounding_rate"],
            "citation_count": cit_scores["count"],
            "tool_calls": len(trace["tool_calls"]),
            "docs_covered": tool_scores.get("docs_covered", []),
            "expected_docs": eq["expected_docs"],
        })

    # ═══ Summary scorecard ════════════════════════════
    hdr("SCORECARD")

    print("\n  Ingestion:")
    for s in ingestion_scores:
        print(f"    {s['doc']}: {s['chunks']} chunks, avg {s['avg_tokens']:.0f} tok, context rate {s['context_rate']:.0%}")

    print("\n  Retrieval (over {} queries):".format(len(retrieval_scores)))
    avg_recall = sum(s["recall"] for s in retrieval_scores) / len(retrieval_scores)
    avg_mrr = sum(s["mrr"] for s in retrieval_scores) / len(retrieval_scores)
    print(f"    Avg Recall:    {avg_recall:.2f}")
    print(f"    Avg MRR:       {avg_mrr:.2f}")

    print("\n  Agentic Chat (over {} questions):".format(len(chat_scores)))
    print(f"    {'Question':<35} {'Tool':>5} {'Calls':>5} {'Cites':>5} {'Ground':>7} {'Docs Found'}")
    print(f"    {'─'*35} {'─'*5} {'─'*5} {'─'*5} {'─'*7} {'─'*20}")
    for s in chat_scores:
        print(f"    {s['description']:<35} {s['tool_use']:>5.2f} {s['tool_calls']:>5} {s['citation_count']:>5} {s['grounding']:>7.0%} {s['docs_covered']}")

    avg_tool = sum(s["tool_use"] for s in chat_scores) / len(chat_scores)
    avg_ground = sum(s["grounding"] for s in chat_scores) / len(chat_scores)
    avg_cites = sum(s["citation_count"] for s in chat_scores) / len(chat_scores)
    total_calls = sum(s["tool_calls"] for s in chat_scores)
    print(f"    {'AVERAGE':<35} {avg_tool:>5.2f} {total_calls/len(chat_scores):>5.1f} {avg_cites:>5.1f} {avg_ground:>7.0%}")

    print(f"\n  Overall:")
    overall = (avg_recall + avg_mrr + avg_tool + avg_ground) / 4
    print(f"    Retrieval Recall:       {avg_recall:.2f}")
    print(f"    Retrieval MRR:          {avg_mrr:.2f}")
    print(f"    Tool Use Effectiveness: {avg_tool:.2f}")
    print(f"    Citation Grounding:     {avg_ground:.2f}")
    print(f"    ─────────────────────────────")
    print(f"    Composite Score:        {overall:.2f}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
