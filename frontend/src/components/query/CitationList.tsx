"use client";

import { Card, CardContent } from "@/components/ui/card";
import { normalizeCitationId, shortCitationId, type Citation } from "@/lib/query";
import { cn } from "@/lib/utils";

export function CitationList({
  citations,
  selectedId,
}: {
  citations: Citation[];
  selectedId: string | null;
}) {
  if (citations.length === 0) return null;
  return (
    <div>
      <div className="mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-[#8a8a8a]">
        Source citations <span className="text-[#111111]">({citations.length})</span>
      </div>
      <Card className="overflow-hidden border-[#111111]/10">
        <CardContent className="p-0">
          {citations.map((citation, index) => {
            const canonical = citation.citation_id
              ? normalizeCitationId(citation.citation_id)
              : `[C${index + 1}]`;
            const isSelected = selectedId === canonical;
            return (
              <div
                // Include the index so a duplicate citation_id in the payload
                // can never produce duplicate React keys.
                key={`${citation.citation_id ?? citation.chunk_id ?? "citation"}-${index}`}
                id={`citation-${shortCitationId(canonical)}`}
                className={cn(
                  "grid scroll-mt-24 grid-cols-[40px_1fr] border-b border-[#111111]/10 transition-colors last:border-b-0",
                  isSelected ? "bg-[#fff5f5]" : "hover:bg-[#fafafa]"
                )}
              >
                <div
                  className={cn(
                    "flex items-start justify-center border-r border-[#111111]/10 px-2 py-4",
                    isSelected ? "bg-[#fdeaea]" : "bg-[#fafafa]"
                  )}
                >
                  <span className="rounded-[4px] border border-[#d62839]/25 bg-white px-1.5 py-0.5 font-mono text-[11px] font-black text-[#d62839]">
                    {canonical}
                  </span>
                </div>
                <div className="p-4">
                  <div className="mb-1 text-[10px] font-bold uppercase tracking-[0.14em] text-[#8a8a8a]">
                    {citation.source ?? citation.section ?? "Document source"}
                    {citation.page_number ? ` · p.${citation.page_number}` : ""}
                  </div>
                  <div className="text-[13px] leading-6 text-[#2a2a2a]">
                    {citation.text || "No excerpt available."}
                  </div>
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}
