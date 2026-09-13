"use client";

import type { HistoryEntry } from "@/lib/query";

export function QueryHistory({
  entries,
  onSelect,
  onClear,
}: {
  entries: HistoryEntry[];
  onSelect: (entry: HistoryEntry) => void;
  onClear: () => void;
}) {
  if (entries.length === 0) return null;
  return (
    <div className="mt-10">
      <div className="mb-3 flex items-center justify-between">
        <div className="micro-label">Recent queries</div>
        <button
          type="button"
          onClick={onClear}
          className="cursor-pointer text-[10px] font-bold uppercase tracking-[0.14em] text-[#8a8a8a] underline-offset-2 transition-colors hover:text-[#d62839] hover:underline focus-visible:outline-2 focus-visible:outline-[#d62839]"
        >
          Clear
        </button>
      </div>
      <div className="space-y-2">
        {entries.map((entry, index) => (
          <button
            key={`${entry.question}-${index}`}
            type="button"
            onClick={() => onSelect(entry)}
            aria-label={`Restore answer for: ${entry.question}`}
            className="w-full cursor-pointer rounded-[6px] border border-[#111111]/10 bg-white p-3 text-left shadow-[0_1px_2px_rgba(17,17,17,0.03)] transition-[border-color,box-shadow,transform] hover:-translate-y-px hover:border-[#d62839]/40 hover:shadow-[0_4px_14px_-8px_rgba(214,40,57,0.4)] focus-visible:outline-2 focus-visible:outline-[#d62839]"
          >
            <div className="line-clamp-2 text-[12px] font-semibold tracking-[-0.02em] text-[#111111]">
              {entry.question}
            </div>
            <div className="mt-1 line-clamp-1 text-[11px] text-[#7a7a7a]">
              {entry.result.answer || "No answer returned."}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
