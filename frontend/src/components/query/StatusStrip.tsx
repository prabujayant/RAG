"use client";

import { Badge } from "@/components/ui/badge";
import { formatLatency, groundingPresentation, type QueryResponse } from "@/lib/query";

export function StatusStrip({ result }: { result: QueryResponse }) {
  const confidence = typeof result.confidence === "number" ? result.confidence : 0;
  const claims = result.claims ?? [];
  const citations = result.citations ?? [];
  const grounding = groundingPresentation(result);

  const metrics: { label: string; value: React.ReactNode }[] = [
    { label: "Status", value: <Badge variant={grounding.tone}>{grounding.label}</Badge> },
    {
      label: "Confidence",
      value: (
        <Badge variant={confidence >= 0.7 ? "success" : confidence >= 0.4 ? "warning" : "danger"}>
          {confidence > 0 ? `${Math.round(confidence * 100)}%` : "n/a"}
        </Badge>
      ),
    },
    { label: "Claims", value: <span className="text-[13px] font-bold text-[#111111]">{claims.length}</span> },
    { label: "Sources", value: <span className="text-[13px] font-bold text-[#111111]">{citations.length}</span> },
  ];

  return (
    <div className="grid grid-cols-2 gap-px border-b border-[#111111]/10 bg-[#111111]/10 sm:grid-cols-4">
      {metrics.map((metric) => (
        <div key={metric.label} className="flex flex-col gap-1.5 bg-white px-4 py-3">
          <span className="text-[9px] font-bold uppercase tracking-[0.16em] text-[#8a8a8a]">
            {metric.label}
          </span>
          {metric.value}
        </div>
      ))}
    </div>
  );
}

export function AnswerMetaFooter({ result }: { result: QueryResponse }) {
  const model = result.model?.trim();
  const latency = formatLatency(result.latency_ms);
  if (!model && latency === "n/a") return null;
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-t border-[#111111]/10 bg-[#fafafa]/60 px-5 py-3 text-[10px] font-medium uppercase tracking-[0.14em] text-[#8a8a8a]">
      {model && (
        <span>
          Model <span className="font-semibold text-[#4a4a4a]">{model}</span>
        </span>
      )}
      {latency !== "n/a" && (
        <span>
          Latency <span className="font-semibold text-[#4a4a4a]">{latency}</span>
        </span>
      )}
    </div>
  );
}
