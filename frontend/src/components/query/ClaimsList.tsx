"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { claimTone, normalizeCitationId, type Claim } from "@/lib/query";

export function ClaimsList({
  claims,
  onCitationSelect,
}: {
  claims: Claim[];
  onCitationSelect: (citationId: string) => void;
}) {
  if (claims.length === 0) return null;
  return (
    <div className="mb-5">
      <div className="mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-[#8a8a8a]">
        Claims &amp; grounding <span className="text-[#111111]">({claims.length})</span>
      </div>
      <Card className="border-[#111111]/10">
        <CardContent className="space-y-3 p-4">
          {claims.map((claim, index) => {
            const text = claim.claim ?? claim.text ?? "Claim unavailable";
            const tone = claimTone(claim.status);
            const linked = claim.citation_ids ?? [];
            return (
              <div
                key={`${text}-${index}`}
                className="grid grid-cols-[24px_1fr] gap-3 rounded-[6px] border border-[#111111]/8 bg-[#fafafa]/50 p-3 last:mb-0"
              >
                <div
                  aria-hidden="true"
                  className={`mt-0.5 flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-black ${
                    tone.badge === "success"
                      ? "bg-[#d6efe4] text-[#1a7a4a]"
                      : tone.badge === "warning"
                        ? "bg-[#fef3c7] text-[#9a6400]"
                        : "bg-[#fdeaea] text-[#d62839]"
                  }`}
                >
                  {tone.glyph}
                </div>
                <div>
                  <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
                    <Badge variant={tone.badge}>{tone.text}</Badge>
                    {linked.map((cid, cidIndex) => (
                      <button
                        key={`${cid}-${cidIndex}`}
                        type="button"
                        onClick={() => onCitationSelect(normalizeCitationId(cid))}
                        aria-label={`Show citation ${normalizeCitationId(cid)} for this claim`}
                        className="cursor-pointer rounded-[4px] border border-[#111111]/15 bg-white px-1.5 py-px font-mono text-[10px] font-bold text-[#5b5b5b] transition-colors hover:border-[#d62839] hover:bg-[#fdeaea] hover:text-[#d62839] focus-visible:outline-2 focus-visible:outline-[#d62839]"
                      >
                        {normalizeCitationId(cid)}
                      </button>
                    ))}
                  </div>
                  <div className="text-[13px] leading-6 text-[#1a1a1a]">{text}</div>
                  {claim.reason && <div className="mt-1 text-[11px] leading-5 text-[#7a7a7a]">{claim.reason}</div>}
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}
