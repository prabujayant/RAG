# AskMyDocs Frontend

Next.js (App Router) UI for the AskMyDocs grounded RAG backend. Proxies the
FastAPI backend at `http://127.0.0.1:8000` (override with `NEXT_PUBLIC_API_URL`).

## Run

```bash
npm install
npm run dev   # http://localhost:3000
```

## What it does

- **Upload** (`UploadPanel`): file input → `POST /api/documents/upload?background=true`
  → polls `GET /api/documents/[id]` until `ready` (elapsed timer while indexing).
- **Ask** (`QueryComposer` → `POST /api/query/stream`): SSE stages
  (`started → retrieval → reranking → evidence → generation → grounding → done`)
  drive the loading labels; the `done` event carries the full `QueryResponse`.
- **Answer** (`AnswerPanel`): grounded badges, confidence bar, per-answer
  **Usage tab** (prompt/completion/total tokens + estimated cost, click to expand),
  `AnswerMetaFooter` (model + latency), clickable `[Cn]` citation chips that
  scroll to `CitationList`, `ClaimsList` with per-claim status, `QueryHistory`.
- **Header** (`SiteHeader`): grounding status dot + cumulative usage pill
  (`Σ tokens · $`, polls `/api/metrics`, click for per-source breakdown).
- **Scope controls**: `AnswerModeToggle` (strict/general) and the
  **shared-corpus checkbox** (default OFF = upload only; ON sends
  `anchor_document_ids` so the corpus is searched while other uploads stay excluded).

## Conventions

- No global state, no data-fetching library: `useState` + `fetch` in `src/app/page.tsx`.
- Backend types mirrored in `src/lib/query.ts` (`QueryResponse`, `TokenUsage`, …);
  keep them in sync with `app/api/schemas/query.py`.
- Proxy routes (`src/app/api/*`) forward bodies verbatim — new backend fields
  (e.g. `usage`) flow through with no proxy changes.
- Styling: Tailwind + local `ui/` primitives (shadcn-style, square corners,
  Swiss-red accent). Check `../AGENTS.md` root for repo-wide guidance.
