from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import structlog
from pydantic_ai import Agent, RunContext

from takehome.config import settings  # noqa: F401 — triggers ANTHROPIC_API_KEY export
from takehome.services.rag import SearchResult, format_search_results, search_chunks

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Dependencies injected into agent tools
# ---------------------------------------------------------------------------


@dataclass
class ChatDeps:
    conversation_id: str
    session: object  # AsyncSession — typed as object to avoid import cycle
    status_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)


# ---------------------------------------------------------------------------
# Citation types
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    doc_label: str
    page: int
    text: str
    section: str | None = None


# ---------------------------------------------------------------------------
# Title-generation agent (lightweight, uses Haiku)
# ---------------------------------------------------------------------------

title_agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    system_prompt="Generate concise conversation titles.",
)


async def generate_title(user_message: str) -> str:
    """Generate a 3-5 word conversation title from the first user message."""
    result = await title_agent.run(
        f"Generate a concise 3-5 word title for a conversation that starts with: '{user_message}'. "
        "Return only the title, nothing else."
    )
    title = str(result.output).strip().strip('"').strip("'")
    if len(title) > 100:
        title = title[:97] + "..."
    return title


# ---------------------------------------------------------------------------
# Chat agent (Sonnet with search_documents tool)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a legal document assistant for commercial real estate lawyers. You help lawyers \
review and understand documents during due diligence.

## Your capabilities
You have access to a search tool that lets you search through uploaded documents. \
Use it to find relevant passages before answering questions.

## How to work
1. When the user asks a question about documents, ALWAYS use the search_documents tool first.
2. You may search multiple times with different queries to find all relevant information.
3. For cross-document analysis, search for the topic across different document contexts.
4. Base your answers strictly on the search results. Do not fabricate information.

## Citation format
When referencing information from documents, you MUST use this exact citation format:
<cite doc="DOC_LABEL" page="PAGE_NUMBER" section="SECTION_OR_CLAUSE">exact or close quote from the document</cite>

For example:
<cite doc="Doc A" page="3" section="Section 1 — Definitions">The Term means a period of fifteen years</cite>
<cite doc="Doc A" page="4" section="3.2 Rent Review">the rent payable shall be reviewed</cite>

Rules for citations:
- Always include the doc label and page number
- Include the section or clause name/number when available from the search results
- The quoted text should be a direct or close paraphrase from the source
- Use multiple citations when drawing from multiple sources
- Every factual claim from a document should have a citation

## Style
- Be concise and precise. Lawyers value accuracy over verbosity.
- Structure longer answers with headings and bullet points.
- When comparing across documents, organize by topic or clause type.
- If information is not found in any document, say so clearly.
"""

chat_agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    system_prompt=SYSTEM_PROMPT,
    deps_type=ChatDeps,
)


@chat_agent.tool  # type: ignore[misc]
async def search_documents(ctx: RunContext[ChatDeps], query: str) -> str:
    """Search uploaded documents for relevant passages.

    Use this tool to find information in the uploaded documents. You can call it
    multiple times with different queries to gather comprehensive information.

    Args:
        query: The search query — be specific about what you're looking for.
    """
    # Push status to the queue so the SSE stream can emit it
    await ctx.deps.status_queue.put(f"Searching: {query}")

    logger.info("Agent searching documents", query=query, conversation_id=ctx.deps.conversation_id)
    results: list[SearchResult] = await search_chunks(
        query=query,
        conversation_id=ctx.deps.conversation_id,
        session=ctx.deps.session,  # type: ignore[arg-type]
        top_k=10,
    )

    # Summarize what was found
    doc_labels = sorted({r.doc_label for r in results})
    summary = f"Found {len(results)} results across {', '.join(doc_labels) if doc_labels else 'no documents'}"
    await ctx.deps.status_queue.put(summary)

    return format_search_results(results)


# ---------------------------------------------------------------------------
# Streaming chat using agent.iter() for proper tool-call support
# ---------------------------------------------------------------------------


async def chat_with_documents(
    user_message: str,
    conversation_history: list[dict[str, str]],
    deps: ChatDeps,
) -> AsyncIterator[str]:
    """Stream a response using the agentic RAG pipeline.

    Uses agent.iter() to properly handle tool calls, then streams the final
    response text. Yields text chunks AND status markers (prefixed with __STATUS__:).
    """
    # Build the prompt with conversation context
    prompt_parts: list[str] = []
    if conversation_history:
        prompt_parts.append("Previous conversation:\n")
        for msg in conversation_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                clean = strip_cite_tags(content)
                prompt_parts.append(f"Assistant: {clean}\n")
        prompt_parts.append("\n")

    prompt_parts.append(f"User: {user_message}")
    full_prompt = "\n".join(prompt_parts)

    # Use agent.iter() to step through the graph, handling tool calls properly
    async with chat_agent.iter(full_prompt, deps=deps) as run:
        async for node in run:
            # Drain status messages from the tool queue
            while not deps.status_queue.empty():
                try:
                    status = deps.status_queue.get_nowait()
                    yield f"__STATUS__:{status}"
                except asyncio.QueueEmpty:
                    break

            # When we hit the final node (agent is about to produce final text),
            # stream it. The final node is a CallToolsNode with only text (no tool calls).
            if hasattr(node, 'model_response'):
                response = node.model_response  # type: ignore[attr-defined]
                # Check if this is the final response (has text parts, no tool-use parts)
                text_parts = [p for p in response.parts if hasattr(p, 'content') and not hasattr(p, 'tool_name')]
                tool_parts = [p for p in response.parts if hasattr(p, 'tool_name')]

                if text_parts and not tool_parts:
                    # This is the final answer — yield the text
                    for part in text_parts:
                        yield part.content

    # Drain any remaining status messages
    while not deps.status_queue.empty():
        try:
            status = deps.status_queue.get_nowait()
            yield f"__STATUS__:{status}"
        except asyncio.QueueEmpty:
            break


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

CITE_RE = re.compile(
    r'<cite\s+doc="([^"]+)"\s+page="(\d+)"(?:\s+section="([^"]*)")?\s*>(.*?)</cite>',
    re.DOTALL,
)


def extract_citations(response: str) -> list[Citation]:
    """Extract structured citations from the response text."""
    citations: list[Citation] = []
    for match in CITE_RE.finditer(response):
        citations.append(
            Citation(
                doc_label=match.group(1),
                page=int(match.group(2)),
                section=match.group(3) if match.group(3) else None,
                text=match.group(4).strip(),
            )
        )
    return citations


def strip_cite_tags(text: str) -> str:
    """Strip <cite> XML tags, leaving just the quoted text with a readable reference."""
    def _replace(m: re.Match[str]) -> str:
        doc = m.group(1)
        page = m.group(2)
        section = m.group(3)
        quoted = m.group(4).strip()
        if section:
            return f'{quoted} [{doc}, {section}, p.{page}]'
        return f'{quoted} [{doc}, p.{page}]'

    return CITE_RE.sub(_replace, text)


def count_sources_cited(response: str) -> int:
    """Count unique citations in the response."""
    return len(extract_citations(response))
