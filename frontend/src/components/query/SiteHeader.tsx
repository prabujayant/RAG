"use client";

import { useEffect, useState } from "react";
import {
  fetchMetricsUsage,
  formatCost,
  formatTokens,
  groundingPresentation,
  type MetricsUsage,
  type QueryResponse,
} from "@/lib/query";

const REFRESH_MS = 20000;

export function SiteHeader({ result }: { result: QueryResponse | null }) {
  const grounding = groundingPresentation(result);
  const dot =
    !result || grounding.label === "Idle"
      ? "bg-[#bfbfbf]"
      : grounding.tone === "success"
        ? "bg-[#1a7a4a]"
        : grounding.tone === "warning"
          ? "bg-[#9a6400]"
          : "bg-[#d62839]";

  const [usage, setUsage] = useState<MetricsUsage | null>(null);
  const [usageOpen, setUsageOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const data = await fetchMetricsUsage();
      if (!cancelled) setUsage(data);
    };
    void load();
    const timer = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // Refresh the cumulative totals right after each answer lands.
  useEffect(() => {
    if (!result) return;
    fetchMetricsUsage().then(setUsage);
  }, [result]);

  const totals = usage?.totals;
  const rows = usage ? Object.entries(usage.by_usage) : [];

  return (
    <header className="sticky top-0 z-30 border-b-2 border-[#111111] bg-[#ffffff]/90 backdrop-blur-sm">
      <div className="mx-auto flex w-full max-w-[1200px] items-stretch justify-between px-4 md:px-8 lg:px-10">
        <div className="flex items-center gap-3 border-r-2 border-[#111111] py-3 pr-5 md:pr-8">
          <div className="flex h-9 w-9 items-center justify-center bg-[#d62839] text-sm font-black text-white">
            Q
          </div>
          <div className="leading-none">
            <div className="text-[15px] font-bold tracking-[-0.05em]">AskMyDocs</div>
            <div className="mt-1 text-[9px] font-medium uppercase tracking-[0.14em] text-[#7a7a7a]">
              Document intelligence
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 py-3 pl-4 md:pl-8">
          <div
            className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-[0.14em] text-[#666666]"
            role="status"
            aria-live="polite"
          >
            <span className={`inline-block h-2.5 w-2.5 rounded-full ${dot}`} aria-hidden="true" />
            {grounding.label}
          </div>
          <div className="relative hidden sm:block">
            <button
              type="button"
              onClick={() => setUsageOpen((v) => !v)}
              aria-expanded={usageOpen}
              aria-label={
                usageOpen ? "Hide cumulative token usage" : "Show cumulative token usage"
              }
              title="Cumulative LLM usage this session (backend process)"
              className="cursor-pointer border border-[#111111]/20 bg-white px-2 py-1 font-mono text-[9px] font-bold uppercase tracking-[0.12em] text-[#111111] transition-colors hover:border-[#d62839] hover:text-[#d62839] focus-visible:outline-2 focus-visible:outline-[#d62839]"
            >
              Σ {totals ? formatTokens(totals.total_tokens) : "…"} ·{" "}
              {totals ? formatCost(totals.cost_usd) : "…"}
            </button>
            {usageOpen && (
              <div className="absolute right-0 top-full z-40 mt-2 w-72 border-2 border-[#111111] bg-white shadow-[4px_4px_0_rgba(17,17,17,0.15)]">
                <div className="border-b border-[#111111]/10 px-3 py-2 text-[9px] font-bold uppercase tracking-[0.16em] text-[#7a7a7a]">
                  Session usage
                </div>
                {rows.length === 0 ? (
                  <div className="px-3 py-3 text-[11px] text-[#666666]">
                    No LLM calls recorded yet — ask a question first.
                  </div>
                ) : (
                  <ul>
                    {rows.map(([label, entry]) => (
                      <li
                        key={label}
                        className="flex items-center justify-between gap-2 border-b border-[#111111]/10 px-3 py-2 font-mono text-[11px] last:border-b-0"
                      >
                        <span className="truncate text-[#5b5b5b]">{label}</span>
                        <span className="shrink-0 font-bold text-[#111111]">
                          {formatTokens(entry.total_tokens)} · {formatCost(entry.cost_usd)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
                {totals && (
                  <div className="flex items-center justify-between gap-2 border-t-2 border-[#111111] bg-[#f5f5f5] px-3 py-2 font-mono text-[11px] font-bold text-[#111111]">
                    <span>Total</span>
                    <span>
                      {formatTokens(totals.total_tokens)} · {formatCost(totals.cost_usd)}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
          <div className="hidden border border-[#d62839] px-2 py-1 text-[9px] font-bold uppercase tracking-[0.16em] text-[#d62839] sm:block">
            RAG · Swiss edition
          </div>
        </div>
      </div>
    </header>
  );
}
