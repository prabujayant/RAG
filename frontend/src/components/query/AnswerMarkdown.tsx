"use client";

import type { ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { withCitationChips } from "./citation-chips";

type SelectFn = (citationId: string) => void;

const PARAGRAPH = "mb-4 text-[15px] leading-7 text-[#111111] last:mb-0";
const HEADING = "mb-3 mt-6 font-black tracking-[-0.04em] text-[#111111] first:mt-0";

function chipped(children: ReactNode, onSelect: SelectFn): ReactNode {
  return withCitationChips(children, onSelect);
}

export function AnswerMarkdown({
  answer,
  onCitationSelect,
}: {
  answer: string;
  onCitationSelect: SelectFn;
}) {
  const components: Components = {
    p: ({ children }) => <p className={PARAGRAPH}>{chipped(children, onCitationSelect)}</p>,
    h1: ({ children }) => <h1 className={`${HEADING} text-xl`}>{chipped(children, onCitationSelect)}</h1>,
    h2: ({ children }) => <h2 className={`${HEADING} text-lg`}>{chipped(children, onCitationSelect)}</h2>,
    h3: ({ children }) => <h3 className={`${HEADING} text-base`}>{chipped(children, onCitationSelect)}</h3>,
    h4: ({ children }) => <h4 className={`${HEADING} text-sm`}>{chipped(children, onCitationSelect)}</h4>,
    ul: ({ children }) => <ul className="mb-4 list-disc space-y-1.5 pl-5 text-[15px] leading-7 text-[#111111]">{children}</ul>,
    ol: ({ children }) => <ol className="mb-4 list-decimal space-y-1.5 pl-5 text-[15px] leading-7 text-[#111111]">{children}</ol>,
    li: ({ children }) => <li>{chipped(children, onCitationSelect)}</li>,
    blockquote: ({ children }) => (
      <blockquote className="mb-4 rounded-r-[6px] border-l-4 border-[#d62839] bg-[#fff2f2] px-4 py-2.5 text-[14px] leading-7 text-[#2a2a2a]">
        {children}
      </blockquote>
    ),
    a: ({ children, href }) => (
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="font-semibold text-[#d62839] underline decoration-[#d62839]/40 underline-offset-2 transition-colors hover:decoration-[#d62839]"
      >
        {children}
      </a>
    ),
    strong: ({ children }) => <strong className="font-bold text-[#111111]">{children}</strong>,
    hr: () => <hr className="my-6 border-t border-[#111111]/12" />,
    table: ({ children }) => (
      <div className="mb-4 overflow-hidden rounded-[6px] border border-[#111111]/12">
        <table className="w-full border-collapse text-[13px] leading-6">{children}</table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-[#111111] text-white">{children}</thead>,
    th: ({ children }) => (
      <th className="border-b border-[#111111]/15 px-3 py-2 text-left text-[11px] font-bold uppercase tracking-[0.08em]">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="border-b border-[#111111]/8 px-3 py-2 align-top text-[#2a2a2a]">
        {chipped(children, onCitationSelect)}
      </td>
    ),
    pre: ({ children }) => (
      <pre className="mb-4 overflow-x-auto rounded-[6px] border border-[#111111] bg-[#151515] p-4 font-mono text-[13px] leading-6 text-[#f5f5f5]">
        {children}
      </pre>
    ),
    code: ({ children, className }) => {
      const isBlock = typeof className === "string" && className.includes("language-");
      if (isBlock) {
        return <code className={className}>{children}</code>;
      }
      return (
        <code className="rounded-[4px] border border-[#111111]/12 bg-[#f3f3f3] px-1.5 py-px font-mono text-[12px] text-[#b5202f]">
          {children}
        </code>
      );
    },
    img: () => null,
  };

  return (
    <div className="answer-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {answer}
      </ReactMarkdown>
    </div>
  );
}
