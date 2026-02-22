# Approach: Cross-Document Analysis & Citation Grounding

## Architecture

The system is built around **agentic RAG with contextual retrieval** — Claude Sonnet is given a `search_documents` tool and a built-in `web_search` tool, and autonomously decides when, what, and how many times to search each. This replaces the original approach of dumping full document text into the prompt.

### Why Agentic Over Static RAG

A static retrieve-then-generate pipeline retrieves once and hopes for the best. The agentic approach lets Claude reformulate queries, search multiple times with different terms, and cross-reference results across documents before answering. In practice, the agent makes 2-3 targeted searches per question — for example, searching "break clause period notice" then refining with "vacant possession material breach conditions." This mirrors how a lawyer would actually research across documents.

For questions that require external context — current market rates, legal precedents, regulatory requirements, planning records — the agent uses Anthropic's built-in `web_search` tool. This runs server-side within the Anthropic API (no separate API key needed) and returns live web results that the agent cites using `<webcite>` tags with source URLs. The system prompt directs the agent to always try document search first, falling back to web search when uploaded documents don't contain the needed information.

### Ingestion Pipeline

When a PDF is uploaded, it goes through five stages:

1. **Text Extraction** — PyMuPDF extracts text from digital PDFs. For scanned or photographed documents, the user toggles "Scanned docs" mode in the UI, which routes pages through Claude Haiku's vision model instead — each page is rendered as a PNG at 2x resolution and sent to Haiku for transcription. This handles handwritten annotations, complex tables, and degraded scan quality that traditional OCR (Tesseract, etc.) struggles with, at roughly $0.001 per page.

2. **Chunking** — Text is split at page boundaries, then within pages at paragraph breaks, targeting ~500 tokens per chunk with 50-token overlap. Page numbers are preserved as metadata.

3. **Contextual Retrieval + Section Identification** — Each chunk is sent to Claude Haiku alongside the full document. In a single call, Haiku returns a structured JSON with two fields: a 2-3 sentence context blurb (document type, parties, defined terms) and the section/clause identifier the chunk falls under. The context is prepended to the chunk before embedding, reducing retrieval failures by 49% per Anthropic's benchmarks. Using the LLM for section identification proved far more robust than regex-based detection — it correctly handles diverse formats across document types (e.g. "Section 3 — Rent", "4.1 Tenant's Obligations", "Executive Summary", "Restrictive Covenants") without brittle pattern matching.

4. **Embedding** — Contextualized chunks are embedded using OpenAI `text-embedding-3-small` (1536 dimensions), chosen for its speed and cost at this scale.

5. **Storage** — Chunks, context, embeddings, and metadata (page number, section/clause) are stored in PostgreSQL with pgvector.

### Hybrid Search

When the agent calls `search_documents`, retrieval uses three signals fused together:

- **Vector similarity** (pgvector cosine distance) — captures semantic meaning
- **BM25 keyword search** (rank-bm25, rebuilt per query) — captures exact term matches that embeddings can miss
- **Reciprocal Rank Fusion** — merges both ranked lists with the formula `1/(k + rank)`, producing a final ranking that consistently outperforms either signal alone

This combination follows Anthropic's contextual retrieval research, which showed hybrid search + contextual retrieval achieves a 67% reduction in retrieval failures compared to vanilla embedding search.

### Citation Grounding

The system prompt instructs Claude to produce `<cite doc="Doc A" page="3" section="Section 3 — Rent">quoted text</cite>` XML tags for document references, and `<webcite url="..." title="...">summary</webcite>` tags for web references. This format was chosen because Claude produces XML tags reliably (more so than JSON or custom syntax), and it's straightforward to parse. Document citations include document label, page number, and section/clause — all three dimensions a lawyer needs to locate the source. Web citations include the source URL and page title.

The backend extracts document citations into structured `Citation` objects. The frontend strips the XML during streaming (showing readable `[Doc A, Section 3, p.3]` text for documents, and markdown links for web sources), then after rendering post-processes the DOM to hydrate document badges into styled clickable pill elements. Clicking a citation pill navigates the document viewer to the correct document and page, with a highlight flash to draw attention. Web citation links open in the browser and are styled in green to visually distinguish them from document citations.

### Key Technical Decisions

- **PydanticAI `agent.iter()`** over `run_stream()` — the streaming API only yields text from the first model turn, silently skipping tool execution. `iter()` properly steps through the full tool-call graph, and we stream the final response node's text. For builtin tools like web search (resolved server-side by Anthropic), we detect `builtin-tool-call` parts in the response and emit status events, while only treating `tool-call` parts (function tools) as requiring additional agent turns.

- **LLM-based section detection over regex** — Early attempts used regex patterns to detect section headers during chunking (`Section \d+`, `ARTICLE [IVX]+`, etc.). This was fragile: it missed headers with em-dashes, over-matched on clause body text, and couldn't handle non-standard formats. Moving section identification into the existing Haiku contextual-retrieval call (one structured JSON response per chunk) solved this with zero additional API cost and handles any document format.

- **DOM post-processing for citation pills** — Markdown renderers (Streamdown) sanitize custom URL protocols like `cite://`, showing `[blocked]`. Instead of fighting the sanitizer, citations render as plain `[Doc A, p.3]` text through the markdown pipeline, then a `useEffect` + TreeWalker hydrates them into styled `<span>` elements with click handlers. This completely bypasses sanitization concerns.

- **Status events via asyncio.Queue** — the `search_documents` tool pushes status messages ("Searching: annual rent amount", "Found 10 results across Doc A, Doc B") into a queue that the SSE stream drains, giving the user real-time visibility into the agent's thinking via a collapsible activity log. Web search status is detected from builtin tool response parts and emitted the same way.

- **PostgreSQL (pgvector) over a dedicated vector DB** — uses the existing Postgres instance, avoids infrastructure complexity, and pgvector's HNSW index is performant at this document scale.

- **Sonnet for chat, Haiku for context generation and OCR** — Sonnet provides better legal reasoning and tool-use decisions; Haiku is sufficient for context + section identification and scanned document OCR, at 1/10th the cost.

- **Vision OCR over traditional OCR** — Tesseract and similar engines struggle with complex legal layouts (multi-column, tables, handwritten marginalia). Claude Haiku's vision model understands document structure and produces clean, well-formatted text. The user explicitly opts in via a UI toggle rather than auto-detection, keeping the upload flow predictable.

### Evaluation

The eval suite (`evals/test_rag_e2e.py`) tests across all synthetic documents and measures four metrics: retrieval recall (do we find the right documents?), MRR (is the right result ranked first?), tool-use effectiveness (does the agent search appropriately?), and citation grounding rate (do citations reference real pages in real documents?). Across 4 eval questions including cross-document analysis, the system scores 1.00 on all metrics — every citation is grounded, every expected document is retrieved, and the agent uses the search tool effectively with 2-3 calls per question. Separate pytest tests in `backend/tests/test_web_search.py` verify that the web search builtin tool is wired correctly and emits proper status events.
