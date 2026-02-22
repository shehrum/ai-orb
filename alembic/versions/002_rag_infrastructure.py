"""RAG infrastructure — document_chunks table, document labels, pgvector

Revision ID: 002_rag
Revises: 001_initial
Create Date: 2025-01-02 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_rag"
down_revision: str = "001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Add label column to documents
    op.add_column("documents", sa.Column("label", sa.String(), nullable=True))

    # Document chunks table — create without embedding column first
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("context_text", sa.Text(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("section_header", sa.String(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
    )

    # Add vector column via raw SQL (pgvector type not supported in sa.Column)
    op.execute("ALTER TABLE document_chunks ADD COLUMN embedding vector(1536);")

    # Index for cosine similarity search (HNSW works on empty tables, unlike IVFFlat)
    op.execute(
        "CREATE INDEX idx_chunks_embedding ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_column("documents", "label")
    op.execute("DROP EXTENSION IF EXISTS vector;")
