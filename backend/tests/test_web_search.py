"""
Test that the web_search builtin tool is wired up correctly on the chat agent
and that status messages are emitted when the agent uses web search.

Usage:
    uv run pytest backend/tests/test_web_search.py -v
"""

from __future__ import annotations

import pytest

from takehome.config import settings  # noqa: F401 — triggers ANTHROPIC_API_KEY export
from takehome.services.llm import (
    ChatDeps,
    chat_agent,
    chat_with_documents,
)

# ---------------------------------------------------------------------------
# Unit-level: agent is configured with WebSearchTool
# ---------------------------------------------------------------------------


def test_agent_has_web_search_builtin():
    """The chat agent should have web_search registered as a builtin tool."""
    kinds = [t.kind for t in chat_agent._builtin_tools]
    assert "web_search" in kinds, f"Expected 'web_search' in builtin tools, got {kinds}"


def test_agent_has_search_documents_tool():
    """The chat agent should still have the custom search_documents tool."""
    tool_names = list(chat_agent._function_toolset.tools.keys())
    assert "search_documents" in tool_names, (
        f"Expected 'search_documents' in tools, got {tool_names}"
    )


# ---------------------------------------------------------------------------
# Integration: agent actually invokes web_search for a web-oriented question
# ---------------------------------------------------------------------------


class FakeSession:
    """Stub that satisfies the session slot in ChatDeps without a real DB."""


@pytest.mark.skipif(not settings.anthropic_api_key, reason="ANTHROPIC_API_KEY not set")
@pytest.mark.asyncio
async def test_web_search_emits_status():
    """Ask a question that should trigger web_search and verify status events."""
    deps = ChatDeps(
        conversation_id="test-web-search",
        session=FakeSession(),
    )

    # A question that clearly requires web context, not uploaded documents
    question = (
        "What are the current average office rental rates per square foot "
        "in the City of London as of 2025?"
    )

    chunks: list[str] = []
    statuses: list[str] = []

    async for chunk in chat_with_documents(question, [], deps):
        if chunk.startswith("__STATUS__:"):
            statuses.append(chunk.removeprefix("__STATUS__:"))
        else:
            chunks.append(chunk)

    response_text = "".join(chunks)

    # The agent should have produced some response
    assert len(response_text) > 0, "Expected non-empty response"

    # At least one status should be a web search
    web_statuses = [s for s in statuses if s.startswith("Web search:")]
    assert len(web_statuses) > 0, (
        f"Expected at least one 'Web search: ...' status message.\n"
        f"All statuses: {statuses}"
    )

    print("\n--- Statuses ---")
    for s in statuses:
        print(f"  {s}")
    print("\n--- Response (first 500 chars) ---")
    print(response_text[:500])
