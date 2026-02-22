# Orbital — AI Engineering Take-Home

Welcome! This is a take-home assessment for an AI engineering role at Orbital.

You've been given a working baseline application: a document Q&A tool for commercial real estate lawyers. Users upload legal documents (leases, title reports, environmental assessments) and ask questions about them. The AI assistant answers questions grounded in the document content.

The app works, but it has limitations. Your job is to extend it.

---

## Setup

### Prerequisites
- Docker and Docker Compose
- just (command runner) — install via `brew install just` or `cargo install just`

That's it. Everything else runs inside containers.

### Getting Started

1. Clone this repository

2. Run the setup command:
```
just setup
```
   This copies `.env.example` to `.env` and builds the Docker images.

3. Add your Anthropic API key to `.env`:
```
ANTHROPIC_API_KEY=your_key_here
```
   We've provided an API key in the task email. You can also use your own.

OPENAI_API_KEY=your_key_here

4. Start everything:
```
just dev
```
   This starts PostgreSQL, the FastAPI backend (port 8000), and the React frontend (port 5173).
   Database migrations run automatically when the backend starts — no separate step needed.

5. Open http://localhost:5173 in your browser.

Your local `backend/src/` and `frontend/src/` directories are mounted into the containers —
edit files normally on your machine and changes hot-reload automatically.

### Sample Documents

We've included two sets of documents:

- `synthetic-docs/` — Programmatically generated legal documents (clean text PDFs). Use these as your primary test case. You can generate more using `scripts/generate-synthetic-docs.py`.
- `real-docs/` — Real-world legal documents including scanned pages and title plans. Use these if you want a harder challenge and visual elments.

### Project Structure

- `frontend/` — React frontend (Vite + Tailwind + shadcn/Radix UI)
- `backend/` — FastAPI backend (Python 3.12 + SQLAlchemy + PydanticAI)
- `alembic/` — Database migrations
- `synthetic-docs/` — Synthetic legal documents for testing
- `real-docs/` — Real-world legal documents (includes scanned/visual content)

### Useful Commands

- `just dev` — Start full stack (Postgres + backend + frontend)
- `just stop` — Stop all services
- `just reset` — Stop everything and clear database
- `just check` — Run all linters and type checks
- `just fmt` — Format all code
- `just db-init` — Run database migrations
- `just db-shell` — Open a psql shell
- `just shell-backend` — Shell into backend container
- `just logs-backend` — Tail backend logs
