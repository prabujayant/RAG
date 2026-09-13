"use client";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { useSpeechRecognition } from "@/lib/useSpeechRecognition";
import { useCallback } from "react";

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

  // Append speech to whatever is already typed, so voice and keyboard compose.
  const handleTranscript = useCallback(
    (text: string) => {
      if (!text) return;
      const base = question.trim();
      onChange(base ? `${base} ${text}` : text);
    },
    [onChange, question]
  );

  const { supported, listening, interim, error, toggle } = useSpeechRecognition({
    onTranscript: handleTranscript,
  });

  return (
    <div className="space-y-3">
      <label className="sr-only" htmlFor="question">
        Question
      </label>
      <div className="relative">
        <Textarea
          id="question"
          value={question}
          onChange={(event) => onChange(event.target.value)}
          placeholder={listening ? "Listening…" : "e.g. How do I rotate API keys?"}
          className={cn(
            "min-h-[84px] border-[#111111]/15 bg-white text-[15px] leading-6 focus-visible:border-[#d62839]",
            listening && "border-[#d62839]/60 pr-12"
          )}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (!submitDisabled) onSubmit();
            }
          }}
        />

        {supported && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={toggle}
            aria-pressed={listening}
            aria-label={listening ? "Stop voice input" : "Start voice input"}
            title={
              listening
                ? "Stop voice input"
                : "Speak your question (uses your browser's speech recognition)"
            }
            className={cn(
              "absolute bottom-2 right-2 h-9 w-9",
              listening
                ? "bg-[#d62839] text-white hover:bg-[#bf2130]"
                : "text-[#8a8a8a] hover:bg-[#111111]/6 hover:text-[#111111]"
            )}
          >
            {listening ? (
              // Stop square: clearer than a filled mic for the active state.
              <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className="size-4">
                <rect x="7" y="7" width="10" height="10" rx="1.5" />
              </svg>
            ) : (
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
                className="size-4"
              >
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v3" />
              </svg>
            )}
          </Button>
        )}
      </div>

      {/* Live feedback while speaking, so the user sees recognition working. */}
      {listening && (
        <p
          className="flex items-center gap-2 text-[12px] italic text-[#6b6b6b]"
          aria-live="polite"
        >
          <span className="inline-block size-1.5 shrink-0 animate-pulse rounded-full bg-[#d62839]" />
          {interim ? interim : "Listening… speak now"}
        </p>
      )}

      {error && (
        <p className="text-[12px] text-[#a81f2d]" role="alert">
          {error}
        </p>
      )}

      <div className="flex items-center justify-between gap-4">
        <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-[#8a8a8a]">
          {disabled
            ? "Upload a document first"
            : supported
              ? "Enter to send · Shift+Enter for a new line · Mic to speak"
              : "Enter to send · Shift+Enter for a new line"}
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
