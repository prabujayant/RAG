"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { formatCost, formatTokens, type QueryResponse } from "@/lib/query";
import { AnswerMarkdown } from "./AnswerMarkdown";
import { AnswerMetaFooter, StatusStrip } from "./StatusStrip";

function UsageTab({ result }: { result: QueryResponse }) {
  const [open, setOpen] = useState(false);
  const usage = result.usage;
  if (!usage) return null;
  const total = usage.total_tokens ?? (usage.prompt_tokens ?? 0) + (usage.completion_tokens ?? 0);
  return (
    <div className="border-t border-[#111111]/10">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={open ? "Hide token usage" : "Show token usage"}
        className="flex w-full cursor-pointer items-center justify-between px-5 py-3 text-[10px] font-bold uppercase tracking-[0.14em] text-[#7a7a7a] transition-colors hover:text-[#d62839] focus-visible:outline-2 focus-visible:outline-[#d62839]"
      >
        <span>
          Usage <span className="text-[#111111]">{formatTokens(total)} tokens</span>
          {" · "}
          <span className="text-[#111111]">{formatCost(usage.cost_usd)}</span>
        </span>
        <span aria-hidden="true">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <dl className="grid grid-cols-2 gap-px border-t border-[#111111]/10 bg-[#111111]/10 md:grid-cols-4">
          {[
            { label: "Prompt", value: formatTokens(usage.prompt_tokens) },
            { label: "Completion", value: formatTokens(usage.completion_tokens) },
            { label: "Total", value: formatTokens(total) },
            { label: "Est. cost", value: formatCost(usage.cost_usd) },
          ].map((row) => (
            <div key={row.label} className="bg-white px-4 py-3">
              <dt className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#7a7a7a]">
                {row.label}
              </dt>
              <dd className="mt-1 font-mono text-[13px] font-bold text-[#111111]">
                {row.value}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

export function AnswerPanel({
  result,
  headingRef,
  onCitationSelect,
}: {
  result: QueryResponse;
  headingRef: React.RefObject<HTMLDivElement | null>;
  onCitationSelect: (citationId: string) => void;
}) {
  const confidence = typeof result.confidence === "number" ? result.confidence : 0;
  const isRefused = result.refused || result.grounding_status === "refused";

  return (
    <>
      <div
        ref={headingRef}
        tabIndex={-1}
        aria-label="Generated answer"
        className="mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-[#8a8a8a] focus-visible:outline-none"
      >
        Generated answer
      </div>

      {isRefused && (
        <div
          role="alert"
          className="mb-5 rounded-[6px] border border-[#d62839]/20 border-l-4 border-l-[#d62839] bg-[#fdeaea] px-4 py-3 text-[13px] leading-6 text-[#a81f2d]"
        >
          <span className="font-bold uppercase tracking-[0.12em]">No grounded answer. </span>
          {result.refused_reason || "The available evidence was not sufficient to answer."}
        </div>
      )}

      {!isRefused && result.generic && (
        <div
          role="note"
          className="mb-5 rounded-[6px] border border-[#1a7a4a]/20 border-l-4 border-l-[#1a7a4a] bg-[#d6efe4]/50 px-4 py-3 text-[13px] leading-6 text-[#14532d]"
        >
          <span className="font-bold uppercase tracking-[0.12em]">General answer. </span>
          Explained freely, with extra context beyond your document. Cited
          statements still come from the document.
        </div>
      )}

      <Card className="mb-5 overflow-hidden border-[#111111]/10">
        <StatusStrip result={result} />
        <CardContent className="p-0">
          <div className="p-6">
            {result.answer ? (
              <AnswerMarkdown answer={result.answer} onCitationSelect={onCitationSelect} />
            ) : (
              <p className="text-[15px] leading-7 text-[#111111]">No answer returned.</p>
            )}
          </div>

          {confidence > 0 && (
            <div className="flex items-center gap-3 border-t border-[#111111]/10 bg-[#fafafa]/60 px-5 py-3 text-[10px] font-bold uppercase tracking-[0.14em] text-[#8a8a8a]">
              <span>Confidence</span>
              <div
                className="h-1.5 w-28 overflow-hidden rounded-full bg-[#111111]/10"
                role="progressbar"
                aria-valuenow={Math.round(confidence * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`Confidence ${Math.round(confidence * 100)} percent`}
              >
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[#e05562] to-[#d62839] transition-[width] duration-700 ease-out"
                  style={{ width: `${Math.max(6, confidence * 100)}%` }}
                />
              </div>
              <span className="text-[#4a4a4a]">{Math.round(confidence * 100)}%</span>
            </div>
          )}

          <AnswerMetaFooter result={result} />
          <UsageTab result={result} />
        </CardContent>
      </Card>
    </>
  );
}
