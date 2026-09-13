"use client";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export function QueryComposer({
  question,
  isLoading,
  disabled = false,
  onChange,
  onSubmit,
}: {
  question: string;
  isLoading: boolean;
  disabled?: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
}) {
  const submitDisabled = isLoading || disabled;

  return (
    <div className="space-y-3">
      <label className="sr-only" htmlFor="question">
        Question
      </label>
      <Textarea
        id="question"
        value={question}
        onChange={(event) => onChange(event.target.value)}
        placeholder="e.g. How do I rotate API keys?"
        className="min-h-[84px] border-[#111111]/15 bg-white text-[15px] leading-6 focus-visible:border-[#d62839]"
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (!submitDisabled) onSubmit();
          }
        }}
      />

      <div className="flex items-center justify-between gap-4">
        <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-[#8a8a8a]">
          {disabled ? "Upload a document first" : "Enter to send · Shift+Enter for a new line"}
        </span>
        <Button
          type="button"
          onClick={onSubmit}
          disabled={submitDisabled}
          className="group min-w-[130px]"
        >
          {isLoading ? "Working…" : "Send"}
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            <path d="M5 12h13M12 5l7 7-7 7" />
          </svg>
        </Button>
      </div>
    </div>
  );
}
