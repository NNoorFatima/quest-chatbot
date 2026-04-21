# FAST University Chatbot

A RAG-based chatbot for FAST National University that answers questions from uploaded PDF documents and the FAST website.

## Features

- **PDF upload** — upload any university document; it gets chunked, embedded, and stored in Supabase
- **Table-aware PDF parsing** — tables are extracted and formatted as markdown so the LLM can read them correctly
- **Query rewriting** — user questions are rewritten into cleaner retrieval queries before hitting the vector store
- **Dual context** — PDF context and website context are both retrieved and sent to the LLM together
- **Streaming responses** — LLM replies stream token-by-token to the browser in real time
- **Website fallback** — if a question matches known topics (admissions, courses, location, contact), the FAST website is scraped for additional context

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| LLM | Groq `llama-3.1-8b-instant` |
| Vector store | Supabase (pgvector) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| PDF parsing | pdfplumber |
| Orchestration | LangChain |

## Supabase Setup

The vector store requires a `documents` table and a `match_documents` RPC function using pgvector. Run the following in your Supabase SQL editor:

```sql
-- Enable pgvector
create extension if not exists vector;

-- Documents table
create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(384)
);

-- Match function used by LangChain SupabaseVectorStore
create or replace function match_documents (
  query_embedding vector(384),
  match_count int default 3
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    id,
    content,
    metadata,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_SERVICE_KEY=your_supabase_service_role_key
   HF_TOKEN=access_token_from_huggingface
   ```

## Running

```bash
uvicorn app.api:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

## RAG Flow

```
Upload PDF
  → pdfplumber extracts text + formats tables as markdown
  → split into 1000-char chunks (200-char overlap)
  → embed with all-MiniLM-L6-v2
  → store in Supabase documents table

User sends message
  → query rewritten by LLM for better retrieval
  → top 3 chunks retrieved from Supabase (pgvector similarity search)
  → website scraped for topic-matched context (admissions / courses / etc.)
  → both contexts passed to Groq Llama 3.1
  → response streamed token-by-token to the browser
```
