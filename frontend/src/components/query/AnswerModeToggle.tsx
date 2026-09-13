"use client";

import { cn } from "@/lib/utils";

export type AnswerMode = "strict" | "general";

const MODES: { value: AnswerMode; label: string; hint: string }[] = [
  {
    value: "strict",
    label: "Strict",
    hint: "Answers only from your document, and refuses when the evidence is not there.",
  },
  {
    value: "general",
    label: "General",
    hint: "Answers freely and explains in more depth, still keeping to your document.",
  },
];

/**
 * Chooses how the model is allowed to answer.
 *
 * Both modes search only the user's uploaded document — the choice changes how
 * the answer is written, not what is retrieved. "General" relaxes the
 * evidence-only contract so the model may add explanatory context instead of
 * refusing, while still grounding cited claims in the document.
 */
export function AnswerModeToggle({
  mode,
  disabled,
  onChange,
}: {
  mode: AnswerMode;
  disabled?: boolean;
  onChange: (mode: AnswerMode) => void;
}) {
  const active = MODES.find((m) => m.value === mode) ?? MODES[0];

  return (
    <div className="mb-4">
      <div
        role="radiogroup"
        aria-label="Answer mode"
        className="inline-flex rounded-[7px] border border-[#111111]/10 bg-[#f4f4f5] p-0.5 shadow-[inset_0_1px_2px_rgba(17,17,17,0.04)]"
      >
        {MODES.map((option) => {
          const selected = option.value === mode;
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={disabled}
              onClick={() => onChange(option.value)}
              className={cn(
                "rounded-[5px] px-3.5 py-1.5 text-[10px] font-bold uppercase tracking-[0.14em] transition-[background-color,color,box-shadow]",
                "focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[#d62839]",
                "disabled:cursor-not-allowed disabled:opacity-45",
                selected
                  ? "bg-white text-[#111111] shadow-[0_1px_2px_rgba(17,17,17,0.12)]"
                  : "bg-transparent text-[#6b6b6b] hover:text-[#111111]"
              )}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      <p className="mt-2 text-[11px] leading-5 text-[#8a8a8a]">{active.hint}</p>
    </div>
  );
}
