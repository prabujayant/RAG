"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { AnswerPanel } from "@/components/query/AnswerPanel";
import { AnswerModeToggle, type AnswerMode } from "@/components/query/AnswerModeToggle";
import { CitationList } from "@/components/query/CitationList";
import { ClaimsList } from "@/components/query/ClaimsList";
import { LoadingPanel } from "@/components/query/LoadingPanel";
import { QueryComposer } from "@/components/query/QueryComposer";
import { QueryHistory } from "@/components/query/QueryHistory";
import { SiteHeader } from "@/components/query/SiteHeader";
import { UploadPanel } from "@/components/query/UploadPanel";
import {
  shortCitationId,
  type HistoryEntry,
  type QueryResponse,
  type UploadedDoc,
} from "@/lib/query";
import { useEffect, useRef, useState } from "react";

const initialQuestion = "";

// Mirrors QueryRequest.question's min_length so a too-short question is caught
// here instead of coming back from the API as an opaque 422.
const MIN_QUESTION_LENGTH = 5;

const LOADING_STAGES = [
  "Starting…",
  "Retrieving evidence…",
  "Reranking…",
  "Generating answer…",
  "Validating claims…",
];

const STREAM_STAGE_INDEX: Record<string, number> = {
  started: 0,
  retrieval: 1,
  retrieved: 1,
  reranking: 2,
  evidence: 2,
  generation: 3,
  grounding: 4,
};

export default function Home() {
  const [question, setQuestion] = useState(initialQuestion);
  const [isLoading, setIsLoading] = useState(false);
  const [stageIndex, setStageIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<string | null>(null);
  const [uploadedDoc, setUploadedDoc] = useState<UploadedDoc | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSeconds, setUploadSeconds] = useState(0);
  // Persisted so the chosen answering style survives a reload. Both modes
  // search only the scoped documents (see includeShared) — this only changes
  // how freely the model may answer (see AnswerModeToggle).
  const [answerMode, setAnswerMode] = useState<AnswerMode>("strict");

  // Retrieval scope. Default OFF: answers are drawn only from the user's own
  // upload. When ON, the shared documentation corpus is searched too — the
  // upload stays anchored (other uploads remain excluded server-side) so an
  // answer never draws specifics from someone else's file.
  const [includeShared, setIncludeShared] = useState(false);
  const hasDocument = Boolean(uploadedDoc);

  // Restore the mode preference across reloads.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      const stored = window.localStorage.getItem("askmydocs.answerMode");
      if (stored === "general" || stored === "strict") setAnswerMode(stored);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  function handleAnswerModeChange(mode: AnswerMode) {
    setAnswerMode(mode);
    window.localStorage.setItem("askmydocs.answerMode", mode);
  }

  const answerHeadingRef = useRef<HTMLDivElement | null>(null);

  // Advance the loading-stage label while a query is in flight.
  useEffect(() => {
    if (!isLoading) return;
    const timer = setInterval(
      () => setStageIndex((index) => Math.min(index + 1, LOADING_STAGES.length - 1)),
      4000
    );
    return () => clearInterval(timer);
  }, [isLoading]);

  // Move focus to the fresh answer (screen readers) and bring it into view.
  useEffect(() => {
    if (!result || isLoading) return;
    answerHeadingRef.current?.focus({ preventScroll: true });
    answerHeadingRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [result, isLoading]);

  function handleCitationSelect(citationId: string) {
    setSelectedCitation(citationId);
    requestAnimationFrame(() => {
      document
        .getElementById(`citation-${shortCitationId(citationId)}`)
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function restoreHistory(entry: HistoryEntry) {
    setQuestion(entry.question);
    setResult(entry.result);
    setSelectedCitation(null);
    setError(null);
  }

  async function submitQuery() {
    const trimmed = question.trim();
    // No document → nothing to search (shared docs are never used).
    if (!trimmed || isLoading || !hasDocument) return;
    if (trimmed.length < MIN_QUESTION_LENGTH) {
      setError(`Please ask at least ${MIN_QUESTION_LENGTH} characters.`);
      return;
    }

    setIsLoading(true);
    setStageIndex(0);
    setError(null);

    try {
      // Prefer the SSE stream so progress renders live; degrade to a plain
      // POST whenever streaming is unavailable or ends without a result.
      try {
        const res = await fetch("/api/query/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
          body: JSON.stringify({
            question: trimmed,
            top_k: 5,
            // Strict filter: the backend searches only this document.
            // Scoped by default (this upload only). With shared docs enabled,
            // the upload stays anchored while the corpus becomes searchable.
            document_ids: includeShared ? undefined : uploadedDoc ? [uploadedDoc.document_id] : undefined,
            anchor_document_ids: includeShared && uploadedDoc ? [uploadedDoc.document_id] : undefined,
            // General mode relaxes the evidence-only contract, but retrieval
            // stays scoped to the upload above.
            allow_generic: answerMode === "general",
          }),
        });
        if (res.ok && res.body) {
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buf = "";
          let donePayload: QueryResponse | null = null;
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            let idx;
            while ((idx = buf.indexOf("\n\n")) !== -1) {
              const raw = buf.slice(0, idx);
              buf = buf.slice(idx + 2);
              const eventMatch = raw.match(/^event:\s*(\w+)/m);
              const dataMatch = raw.match(/^data:\s*([\s\S]*)/m);
              const event = eventMatch?.[1];
              if (event && event in STREAM_STAGE_INDEX) {
                setStageIndex(STREAM_STAGE_INDEX[event]);
              }
              if (event === "done" && dataMatch?.[1]) {
                try {
                  donePayload = JSON.parse(dataMatch[1]) as QueryResponse;
                } catch {
                  donePayload = null;
                }
              } else if (event === "error") {
                throw new Error(dataMatch?.[1] ?? "Request failed");
              }
            }
          }
          if (donePayload) {
            const next: QueryResponse = donePayload;
            setResult(next);
            setSelectedCitation(null);
            setHistory((prev) => [{ question: trimmed, result: next }, ...prev].slice(0, 6));
            return;
          }
          console.warn("Query stream ended without a result; retrying without streaming.");
        }
      } catch (streamErr) {
        // A stream failure is recoverable — fall through to the plain POST
        // rather than surfacing a confusing error to the user.
        console.warn("Query stream failed; retrying without streaming:", streamErr);
      }

      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          top_k: 5,
          document_ids: uploadedDoc ? [uploadedDoc.document_id] : undefined,
          allow_generic: answerMode === "general",
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail?.message || data?.error?.message || "Request failed");
      }

      const next: QueryResponse = data;
      setResult(next);
      setSelectedCitation(null);
      setHistory((prev) => [{ question: trimmed, result: next }, ...prev].slice(0, 6));
    } catch (err) {
      // Clear the previous answer: leaving it on screen next to a failed
      // question makes it look like the answer to *that* question.
      setResult(null);
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setIsLoading(false);
    }
  }

  async function uploadDocument(file: File) {
    setIsUploading(true);
    setUploadError(null);
    setUploadSeconds(0);

    // Background ingest: the API returns 201/processing immediately while the
    // Celery worker embeds + indexes (~20-60s cold). Poll document status
    // until READY instead of holding one HTTP request open.
    const startedAt = Date.now();
    const timer = setInterval(
      () => setUploadSeconds(Math.round((Date.now() - startedAt) / 1000)),
      500
    );
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10 * 60 * 1000);

    async function pollUntilReady(documentId: string): Promise<{ title: string; chunk_count: number | null }> {
      for (let i = 0; i < 120; i++) {
        await new Promise((r) => setTimeout(r, 2000));
        const st = await fetch(`/api/documents/${encodeURIComponent(documentId)}`, {
          signal: controller.signal,
        });
        if (!st.ok) continue;
        const doc = await st.json();
        if (doc?.status === "ready" || doc?.status === "failed") return doc;
      }
      throw new Error("Ingest timed out waiting for the worker.");
    }

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch("/api/documents/upload?background=true", {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail?.message || "Upload failed");
      }

      if (data?.status === "processing") {
        const doc = await pollUntilReady(data.document_id);
        if (doc && (doc as { status?: string }).status === "failed") {
          throw new Error("Ingestion failed on the worker.");
        }
        setUploadedDoc({
          document_id: data.document_id,
          title: data.title,
          chunk_count: (doc as { chunk_count?: number | null }).chunk_count ?? null,
        });
      } else {
        setUploadedDoc({
          document_id: data.document_id,
          title: data.title,
          chunk_count: data.chunk_count,
        });
      }
      setResult(null);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setUploadError("Upload timed out after 10 minutes.");
      } else {
        setUploadError(err instanceof Error ? err.message : "Unexpected error");
      }
    } finally {
      clearInterval(timer);
      clearTimeout(timeout);
      setIsUploading(false);
    }
  }

  const claims = result?.claims ?? [];
  const citations = result?.citations ?? [];

  return (
    <div className="min-h-screen text-[#111111]">
      <SiteHeader result={result} />

      <main className="mx-auto grid w-full max-w-[1200px] gap-8 px-4 py-8 md:grid-cols-[300px_minmax(0,1fr)] md:gap-10 md:px-8 lg:px-10 lg:py-12">
        <aside className="md:pt-2">
          <div className="mb-6">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[#d62839]/25 bg-[#fdeaea]/60 px-3 py-1 text-[10px] font-bold uppercase tracking-[0.18em] text-[#d62839]">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-[#d62839]" aria-hidden="true" />
              Prompt
            </div>
            <h1 className="text-[34px] font-black leading-[1.02] tracking-[-0.055em]">
              Ask a question.
            </h1>
          </div>

          <p className="mb-6 text-[13px] leading-6 text-[#5b5b5b]">
            Ask questions about your uploaded document and inspect the grounded evidence behind every claim.
          </p>

          <UploadPanel
            isUploading={isUploading}
            uploadSeconds={uploadSeconds}
            uploadError={uploadError}
            uploadedDoc={uploadedDoc}
            onFileSelect={(file) => void uploadDocument(file)}
          />

          {!hasDocument && (
            <div className="mb-4 flex gap-2.5 rounded-[6px] border-l-4 border-[#111111]/15 bg-[#f4f4f5] px-3 py-2.5 text-[12px] leading-5 text-[#5b5b5b]">
              <span>
                Upload a document to ask questions. Answers are drawn only from the document you provide.
              </span>
            </div>
          )}

          <AnswerModeToggle
            mode={answerMode}
            disabled={!hasDocument}
            onChange={handleAnswerModeChange}
          />

          <label className="mb-4 flex cursor-pointer items-start gap-2 text-[12px] leading-5 text-[#5b5b5b]">
            <input
              type="checkbox"
              checked={includeShared}
              disabled={!hasDocument}
              onChange={(event) => setIncludeShared(event.target.checked)}
              className="mt-1 accent-[#d62839] disabled:cursor-not-allowed"
            />
            <span>
              Include shared documentation.{" "}
              <span className="text-[#7a7a7a]">
                {includeShared
                  ? "Answers may cite the shared corpus as well as your upload."
                  : "Answers are drawn only from your upload."}
              </span>
            </span>
          </label>

          <QueryComposer
            question={question}
            isLoading={isLoading}
            disabled={!hasDocument}
            onChange={setQuestion}
            onSubmit={() => void submitQuery()}
          />

          {error && (
            <div
              role="alert"
              className="mt-4 rounded-[6px] border-l-4 border-[#d62839] bg-[#fdeaea] px-3 py-2.5 text-[13px] leading-5 text-[#a81f2d]"
            >
              {error}
            </div>
          )}

          <QueryHistory
            entries={history}
            onSelect={restoreHistory}
            onClear={() => setHistory([])}
          />
        </aside>

        <section className="min-w-0" aria-label="Answer">
          {isLoading ? (
            <LoadingPanel stage={LOADING_STAGES[stageIndex]} />
          ) : !result ? (
            <Card className="overflow-hidden border-[#111111]/10">
              <CardHeader className="border-b border-[#111111]/10 bg-[#fafafa]/60 pb-4">
                <div className="flex items-center justify-between gap-3">
                  <span className="micro-label">Response</span>
                  <Badge variant="secondary">Waiting</Badge>
                </div>
              </CardHeader>
              <CardContent className="flex flex-col items-center justify-center px-6 py-20 text-center">
                <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-full border border-[#d62839]/20 bg-[#fdeaea] text-[#d62839]">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                    className="h-6 w-6"
                  >
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <path d="M14 2v6h6" />
                    <path d="M9 13h6M9 17h4" />
                  </svg>
                </div>
                <h2 className="text-lg font-black tracking-[-0.04em]">No answer yet</h2>
                <p className="mt-2 max-w-sm text-[13px] leading-6 text-[#5b5b5b]">
                  Upload a document and ask a question to inspect a grounded answer, per-claim
                  validation, and supporting citations.
                </p>
                <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
                  {["Grounded answers", "Per-claim checks", "Source citations"].map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-[#111111]/10 bg-white px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#6b6b6b]"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="animate-fade-up">
              <AnswerPanel
                result={result}
                headingRef={answerHeadingRef}
                onCitationSelect={handleCitationSelect}
              />
              <ClaimsList claims={claims} onCitationSelect={handleCitationSelect} />
              <CitationList citations={citations} selectedId={selectedCitation} />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
