from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.session import get_session
from takehome.services.conversation import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    update_conversation,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class ConversationListItem(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    has_document: bool

    model_config = {"from_attributes": True}


class DocumentInfo(BaseModel):
    id: str
    filename: str
    page_count: int
    label: str | None = None
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    has_document: bool
    documents: list[DocumentInfo] = []

    model_config = {"from_attributes": True}


class ConversationCreate(BaseModel):
    pass


class ConversationUpdate(BaseModel):
    title: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get("", response_model=list[ConversationListItem])
async def list_conversations_endpoint(
    session: AsyncSession = Depends(get_session),
) -> list[ConversationListItem]:
    """List all conversations, ordered by most recently updated."""
    conversations = await list_conversations(session)
    return [
        ConversationListItem(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            has_document=len(c.documents) > 0,
        )
        for c in conversations
    ]


@router.post("", response_model=ConversationDetail, status_code=201)
async def create_conversation_endpoint(
    session: AsyncSession = Depends(get_session),
) -> ConversationDetail:
    """Create a new conversation."""
    conversation = await create_conversation(session)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        has_document=False,
        documents=[],
    )


def _build_detail(conversation) -> ConversationDetail:  # type: ignore[no-untyped-def]
    """Helper to build ConversationDetail from a Conversation ORM object."""
    docs = [
        DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            page_count=doc.page_count,
            label=doc.label,
            uploaded_at=doc.uploaded_at,
        )
        for doc in conversation.documents
    ]
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        has_document=len(docs) > 0,
        documents=docs,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_endpoint(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
) -> ConversationDetail:
    """Get a single conversation with its documents."""
    conversation = await get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _build_detail(conversation)


@router.patch("/{conversation_id}", response_model=ConversationDetail)
async def update_conversation_endpoint(
    conversation_id: str,
    body: ConversationUpdate,
    session: AsyncSession = Depends(get_session),
) -> ConversationDetail:
    """Update a conversation's title."""
    conversation = await update_conversation(session, conversation_id, body.title)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _build_detail(conversation)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation_endpoint(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a conversation and all associated data."""
    deleted = await delete_conversation(session, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
