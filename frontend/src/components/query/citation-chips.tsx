"use client";

import type { ReactNode } from "react";
import { Children, cloneElement, isValidElement } from "react";
import { CITATION_MARK_RE, normalizeCitationId, shortCitationId } from "@/lib/query";

/** Non-global single-marker test (avoids shared lastIndex state). */
const SINGLE_MARK_RE = /^\[C\d+\]$/;

export function CitationChip({
  citationId,
  onSelect,
}: {
  citationId: string;
  onSelect: (citationId: string) => void;
}) {
  const canonical = normalizeCitationId(citationId);
  return (
    <button
      type="button"
      onClick={() => onSelect(canonical)}
      aria-label={`Show citation ${shortCitationId(canonical)}`}
      title={`Show citation ${canonical}`}
      className="mx-0.5 inline-flex -translate-y-px cursor-pointer items-center rounded-[4px] border border-[#d62839]/35 bg-[#fdeaea] px-1.5 py-px align-baseline font-mono text-[11px] font-bold text-[#d62839] transition-[background-color,color,box-shadow] hover:bg-[#d62839] hover:text-white hover:shadow-[0_2px_8px_-3px_rgba(214,40,57,0.7)] focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[#d62839]"
    >
      {canonical}
    </button>
  );
}

/**
 * Recursively walk rendered children and replace "[Cn]" text runs with
 * clickable CitationChip buttons. Used inside markdown block components so
 * inline answer markers link to the citation cards below.
 */
export function withCitationChips(
  children: ReactNode,
  onSelect: (citationId: string) => void
): ReactNode {
  return Children.map(children, (child, index) => {
    if (typeof child === "string") {
      const parts = child.split(CITATION_MARK_RE);
      if (parts.length === 1) return child;
      return parts.map((part, partIndex) => {
        const key = `${index}-${partIndex}`;
        if (SINGLE_MARK_RE.test(part)) {
          return <CitationChip key={key} citationId={part} onSelect={onSelect} />;
        }
        return <span key={key}>{part}</span>;
      });
    }
    if (isValidElement<{ children?: ReactNode }>(child) && child.props.children) {
      // Recurse into nested inline elements (strong, em, a, …).
      const nested = withCitationChips(child.props.children, onSelect);
      if (nested !== child.props.children) {
        return cloneElement(child, { ...child.props, children: nested });
      }
    }
    return child;
  });
}
