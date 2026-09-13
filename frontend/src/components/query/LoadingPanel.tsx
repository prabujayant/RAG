"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";

export function LoadingPanel({ stage }: { stage: string }) {
  return (
    <Card className="overflow-hidden border-[#111111]/10" aria-busy="true">
      <CardHeader className="border-b border-[#111111]/10 bg-[#fafafa]/60 pb-4">
        <div className="flex items-center justify-between gap-3">
          <span className="micro-label">Response</span>
          <span
            role="status"
            aria-live="polite"
            className="inline-flex items-center gap-2 rounded-full border border-[#d62839]/25 bg-[#fff2f2] px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#d62839]"
          >
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#d62839]" />
            {stage}
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 p-6" aria-hidden="true">
        {["w-11/12", "w-full", "w-4/5", "w-3/5"].map((width, index) => (
          <div
            key={width}
            className={`h-3.5 animate-pulse rounded-full bg-[#111111]/8 ${width}`}
            style={{ animationDelay: `${index * 120}ms` }}
          />
        ))}
      </CardContent>
    </Card>
  );
}
