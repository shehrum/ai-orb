# Approach

## Architecture

The system is an **agentic RAG pipeline** — Claude Sonnet is given two tools (`search_documents` for uploaded PDFs, `web_search` for live web context) and autonomously decides when to use each. Rather than retrieving once and hoping for the best, the agent reformulates queries, searches multiple times, and cross-references results across documents before answering. In practice it makes 2-3 targeted searches per question, mirroring how a lawyer actually researches.

**Ingestion** follows five stages: text extraction (PyMuPDF for digital PDFs, Claude Haiku vision for scanned docs), chunking at page/paragraph boundaries (~500 tokens), contextual retrieval + section identification via Haiku (a single structured JSON call per chunk that adds a context blurb and identifies the clause/section), embedding with OpenAI `text-embedding-3-small`, and storage in PostgreSQL with pgvector.

**Retrieval** fuses three signals: vector similarity (pgvector cosine distance), BM25 keyword search, and Reciprocal Rank Fusion to merge both ranked lists. This hybrid approach, combined with contextual retrieval, achieves a 67% reduction in retrieval failures compared to vanilla embedding search per Anthropic's benchmarks.

**Citations** use XML tags that Claude produces reliably. Document references (`<cite doc="..." page="..." section="...">`) carry doc label, page, and clause — the three dimensions a lawyer needs to locate the source. Web references (`<webcite url="..." title="...">`) carry the source URL. The frontend hydrates both into styled clickable pills via DOM post-processing after the markdown renderer finishes, sidestepping sanitization issues with custom URL protocols.

**Web search** uses Anthropic's built-in server-side tool (no extra API key), letting the agent fetch current market rates, legal precedents, and regulatory info when the uploaded documents don't have the answer.

## What I Prioritised

1. **Retrieval quality** — hybrid search + contextual retrieval because the answer is only as good as what the model sees. The eval suite confirms 1.00 recall and MRR across all test questions.
2. **Citation grounding** — every factual claim traces back to a specific page and section. Clickable pills navigate the document viewer directly to the source. The eval confirms 100% grounding rate.
3. **Agentic tool use** — letting the model decide search strategy rather than hard-coding retrieval logic. This handles cross-document questions naturally (the agent searches each topic separately and synthesises).
4. **Real-time UX** — SSE streaming with a collapsible activity log showing each search step as it happens, with distinct icons for document vs web searches.

## What I'd Do Next

- **Streaming for web search responses** — the builtin web search tool returns the full response in one shot rather than streaming token-by-token. Chunking it into ~80-char SSE pieces helps, but true incremental streaming would feel smoother.
- **Conversation-aware retrieval** — currently each question is searched independently. Incorporating conversation history into the search query would handle follow-up questions ("what about the break clause?" after discussing rent) without the user restating context.
- **Table-aware chunking** — the current paragraph-boundary chunking can split tables across chunks. Detecting table boundaries during extraction and keeping them whole would improve retrieval for tabular data like rent schedules.
- **Evaluation breadth** — the eval suite covers 4 questions across 3 documents. A production system needs adversarial questions (ambiguous, unanswerable, contradictory across documents) and a larger document corpus.

## Interesting Problems

**PydanticAI's `agent.iter()` vs builtin tools** — the streaming loop checks response parts for tool calls to decide whether to yield text or loop for another agent turn. When Anthropic's server-side web search was added, the response contained both `builtin-tool-call` parts and `text` parts in a single response. The original check (`hasattr(p, 'tool_name')`) treated all tool parts equally, which blocked text extraction when web search was used. The fix was switching to `part_kind` discriminators: only `tool-call` (function tools needing local execution) triggers another turn, while `builtin-tool-call` (already resolved server-side) doesn't.

**Citation pills vs markdown sanitisation** — Streamdown (the markdown renderer) sanitises custom URL protocols, so `cite://DocA/page/3` renders as `[blocked]`. Rather than fighting the sanitiser, citations render as plain text badges (`[Doc A, Section 3, p.3]`) through the markdown pipeline, then a `useEffect` + TreeWalker hydrates them into styled interactive elements. The same approach works for web citations (`[Web · title]` badges hydrated into `<a>` pill elements).

**Status event timing for builtin tools** — web search status events and the full response text arrived in the same SSE read batch, so the frontend never rendered the "searching" state. Adding a 50ms flush delay after status events and chunking large responses into smaller SSE pieces ensures the browser renders the activity log before content starts appearing.
