export type TokenUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  cost_usd?: number | null;
};

export type Citation = {
  citation_id?: string;
  chunk_id?: string;
  text?: string;
  page_number?: number | null;
  section?: string | null;
  source?: string;
};

export type Claim = {
  claim?: string;
  text?: string;
  status?: string;
  reason?: string | null;
  citation_ids?: string[];
};

export type QueryResponse = {
  answer?: string;
  grounded?: boolean;
  grounding_status?: string;
  confidence?: number | null;
  citations?: Citation[];
  claims?: Claim[];
  refused?: boolean;
  refused_reason?: string | null;
  latency_ms?: number | null;
  model?: string | null;
  usage?: TokenUsage | null;
  generic?: boolean;
};

export type UploadedDoc = {
  document_id: string;
  title: string;
  chunk_count?: number | null;
};

export type HistoryEntry = {
  question: string;
  result: QueryResponse;
};

/** Matches inline citation markers like [C1], [C12]. */
export const CITATION_MARK_RE = /(\[C\d+\])/g;

/** Canonical display form: "[C1]". Accepts "C1" or "[C1]". */
export function normalizeCitationId(raw: string): string {
  const m = raw.match(/C\d+/i);
  return m ? `[${m[0].toUpperCase()}]` : raw;
}

/** "C1" without brackets — used for element ids and aria labels. */
export function shortCitationId(raw: string): string {
  return normalizeCitationId(raw).replace(/[\[\]]/g, "");
}
export function formatLatency(ms: number | null | undefined): string {
  if (typeof ms !== "number" || Number.isNaN(ms) || ms < 0) return "n/a";
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

export function formatTokens(n: number | null | undefined): string {
  if (typeof n !== "number" || Number.isNaN(n) || n < 0) return "n/a";
  return Math.round(n).toLocaleString("en-US");
}

export function formatCost(usd: number | null | undefined): string {
  if (typeof usd !== "number" || Number.isNaN(usd) || usd < 0) return "n/a";
  if (usd === 0) return "$0.00";
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

export type UsageTotals = {
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
};

export type MetricsUsage = {
  by_usage: Record<string, UsageTotals>;
  totals: UsageTotals;
};

export async function fetchMetricsUsage(): Promise<MetricsUsage | null> {
  try {
    const res = await fetch("/api/metrics", { cache: "no-store" });
    if (!res.ok) return null;
    const data = await res.json();
    return (data?.llm_usage as MetricsUsage) ?? null;
  } catch {
    return null;
  }
}

export type GroundingTone = "success" | "warning" | "danger";

export function groundingPresentation(result: QueryResponse | null): {
  label: string;
  tone: GroundingTone;
} {
  if (!result) return { label: "Idle", tone: "success" };
  if (result.refused || result.grounding_status === "refused") {
    return { label: "Refused", tone: "danger" };
  }
  if (result.generic) {
    return { label: "General answer", tone: "warning" };
  }
  switch (result.grounding_status) {
    case "grounded":
      return { label: "Grounded", tone: "success" };
    case "partially_grounded":
      return { label: "Partially grounded", tone: "warning" };
    case "ungrounded":
      return { label: "Ungrounded", tone: "danger" };
    default:
      return result.grounded
        ? { label: "Grounded", tone: "success" }
        : { label: "Needs review", tone: "danger" };
  }
}

export function claimTone(status: string | undefined): {
  badge: "success" | "warning" | "danger";
  text: "supported" | "partial" | "unsupported";
  glyph: "✓" | "?" | "×";
} {
  if (status === "supported") return { badge: "success", text: "supported", glyph: "✓" };
  if (status === "partially_supported" || status === "unknown" || status === undefined)
    return { badge: "warning", text: "partial", glyph: "?" };
  return { badge: "danger", text: "unsupported", glyph: "×" };
}
