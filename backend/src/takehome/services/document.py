from __future__ import annotations

import base64
import os
import uuid

import anthropic
import fitz  # PyMuPDF
import structlog
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.config import settings
from takehome.db.models import Document

logger = structlog.get_logger()

# Label sequence: "Doc A", "Doc B", ..., "Doc Z", "Doc AA", ...
_LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def _label_for_index(idx: int) -> str:
    """Generate a document label for the given 0-based index."""
    if idx < 26:
        return f"Doc {_LABELS[idx]}"
    return f"Doc {_LABELS[idx // 26 - 1]}{_LABELS[idx % 26]}"


def _extract_text_pymupdf(file_path: str) -> tuple[str, int]:
    """Extract text using PyMuPDF. Returns (text, page_count)."""
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        pages: list[str] = []
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()  # type: ignore[union-attr]
            if text.strip():
                pages.append(f"--- Page {page_num + 1} ---\n{text}")
        doc.close()
        return "\n\n".join(pages), page_count
    except Exception:
        logger.exception("Failed to extract text from PDF", path=file_path)
        return "", 0


async def _ocr_with_vision(file_path: str) -> tuple[str, int]:
    """OCR a scanned PDF using Claude Haiku vision.

    Renders each page as an image and sends it to Claude for transcription.
    Returns (text, page_count).
    """
    client = anthropic.AsyncAnthropic()
    doc = fitz.open(file_path)
    page_count = len(doc)
    pages: list[str] = []

    for page_num in range(page_count):
        page = doc[page_num]

        # Render page to PNG at 2x resolution for quality
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)  # type: ignore[union-attr]
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        try:
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe all the text on this page of a legal document. "
                                    "Preserve the structure: section headings, clause numbers, "
                                    "paragraph breaks, tables, and any handwritten annotations. "
                                    "Return only the transcribed text, nothing else."
                                ),
                            },
                        ],
                    }
                ],
            )
            page_text = response.content[0].text  # type: ignore[union-attr]
            pages.append(f"--- Page {page_num + 1} ---\n{page_text}")
            logger.info("OCR completed for page", page=page_num + 1, chars=len(page_text))
        except Exception:
            logger.exception("OCR failed for page", page=page_num + 1)
            pages.append(f"--- Page {page_num + 1} ---\n[OCR failed for this page]")

    doc.close()
    return "\n\n".join(pages), page_count


async def upload_document(
    session: AsyncSession, conversation_id: str, file: UploadFile, *, use_ocr: bool = False
) -> Document:
    """Upload and process a PDF document for a conversation.

    When use_ocr=True, pages are rendered as images and sent to Claude Haiku
    vision for transcription (for scanned/photographed documents).
    """
    # Validate file type
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > settings.max_upload_size:
        raise ValueError(
            f"File too large. Maximum size is {settings.max_upload_size // (1024 * 1024)}MB."
        )

    # Generate a unique filename to avoid collisions
    original_filename = file.filename or "document.pdf"
    unique_name = f"{uuid.uuid4().hex}_{original_filename}"
    file_path = os.path.join(settings.upload_dir, unique_name)

    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Save the file to disk
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info("Saved uploaded PDF", filename=original_filename, path=file_path, size=len(content))

    # Extract text
    if use_ocr:
        logger.info("Using Vision OCR for scanned document", filename=original_filename)
        extracted_text, page_count = await _ocr_with_vision(file_path)
    else:
        extracted_text, page_count = _extract_text_pymupdf(file_path)

    logger.info(
        "Text extraction complete",
        filename=original_filename,
        page_count=page_count,
        text_length=len(extracted_text),
        method="ocr" if use_ocr else "pymupdf",
    )

    # Determine label: count existing documents in this conversation
    count_stmt = select(func.count()).where(Document.conversation_id == conversation_id)
    result = await session.execute(count_stmt)
    existing_count = result.scalar() or 0
    label = _label_for_index(existing_count)

    # Create the document record
    document = Document(
        conversation_id=conversation_id,
        filename=original_filename,
        file_path=file_path,
        extracted_text=extracted_text if extracted_text else None,
        page_count=page_count,
        label=label,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    # Trigger RAG processing (chunk + embed)
    try:
        from takehome.services.rag import process_document

        await process_document(session, document)
        logger.info("RAG processing complete", document_id=document.id, label=label)
    except Exception:
        logger.exception("RAG processing failed", document_id=document.id)

    return document


async def get_document(session: AsyncSession, document_id: str) -> Document | None:
    """Get a document by its ID."""
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_document_for_conversation(
    session: AsyncSession, conversation_id: str
) -> Document | None:
    """Get the first document for a conversation, if one exists."""
    stmt = select(Document).where(Document.conversation_id == conversation_id)
    result = await session.execute(stmt)
    return result.scalars().first()


async def get_documents_for_conversation(
    session: AsyncSession, conversation_id: str
) -> list[Document]:
    """Get all documents for a conversation, ordered by upload time."""
    stmt = (
        select(Document)
        .where(Document.conversation_id == conversation_id)
        .order_by(Document.uploaded_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
