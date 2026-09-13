"use client";

import { useRef, useState } from "react";
import type { UploadedDoc } from "@/lib/query";
import { cn } from "@/lib/utils";

export function UploadPanel({
  isUploading,
  uploadSeconds,
  uploadError,
  uploadedDoc,
  onFileSelect,
}: {
  isUploading: boolean;
  uploadSeconds: number;
  uploadError: string | null;
  uploadedDoc: UploadedDoc | null;
  onFileSelect: (file: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  function pickFile(file: File | undefined | null) {
    if (file) onFileSelect(file);
  }

  return (
    <div className="mb-6 overflow-hidden rounded-[var(--radius)] border border-[#111111]/10 bg-white shadow-[var(--shadow-card)]">
      <div className="flex items-center justify-between border-b border-[#111111]/10 px-4 py-3">
        <div className="micro-label">Your document</div>
        {uploadedDoc && !isUploading && (
          <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#15683e]">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-[#1a7a4a]" aria-hidden="true" />
            Ready
          </span>
        )}
        {isUploading && (
          <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#d62839]">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[#d62839]" aria-hidden="true" />
            Indexing
          </span>
        )}
      </div>

      <div className="p-3">
        <input
          ref={inputRef}
          id="doc-upload"
          type="file"
          accept=".pdf,.docx,.html,.htm,.md"
          disabled={isUploading}
          onChange={(event) => {
            pickFile(event.target.files?.[0]);
            event.target.value = "";
          }}
          className="sr-only"
        />

        <div
          role="button"
          tabIndex={0}
          aria-label="Choose a document to upload"
          aria-disabled={isUploading}
          onClick={() => !isUploading && inputRef.current?.click()}
          onKeyDown={(event) => {
            if (!isUploading && (event.key === "Enter" || event.key === " ")) {
              event.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragOver={(event) => {
            event.preventDefault();
            if (!isUploading) setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            if (!isUploading) pickFile(event.dataTransfer.files?.[0]);
          }}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-[6px] border border-dashed px-4 py-5 text-center transition-colors",
            isUploading && "cursor-not-allowed opacity-60",
            isDragging
              ? "border-[#d62839] bg-[#fdeaea]/60"
              : "border-[#111111]/20 bg-[#fafafa] hover:border-[#d62839]/50 hover:bg-[#fff7f7]"
          )}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            className={cn("h-5 w-5 transition-colors", isDragging ? "text-[#d62839]" : "text-[#8a8a8a]")}
          >
            <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
            <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
          </svg>
          <div className="text-[12px] font-semibold text-[#111111]">
            {isDragging ? "Drop to upload" : isUploading ? "Uploading…" : "Choose a file or drop it here"}
          </div>
          <div className="text-[10px] uppercase tracking-[0.14em] text-[#8a8a8a]">
            PDF · DOCX · HTML · Markdown
          </div>
        </div>

        {isUploading && (
          <div className="mt-3 text-[11px] text-[#666666]" role="status">
            Parsing, embedding and indexing… {uploadSeconds}s elapsed
            <div className="mt-0.5 text-[10px] text-[#999999]">
              The first upload can take about a minute while the embedding model loads.
            </div>
          </div>
        )}

        {uploadError && (
          <div className="mt-3 rounded-[4px] border-l-4 border-[#d62839] bg-[#fdeaea] px-3 py-2 text-[11px] leading-5 text-[#a81f2d]">
            {uploadError}
          </div>
        )}

        {uploadedDoc && !isUploading && (
          <div className="mt-3 rounded-[6px] border border-[#111111]/10 bg-[#fafafa] px-3 py-2.5">
            <div className="truncate text-[12px] font-semibold text-[#111111]">{uploadedDoc.title}</div>
            <div className="mt-0.5 text-[10px] uppercase tracking-[0.14em] text-[#8a8a8a]">
              {uploadedDoc.chunk_count ?? 0} chunks indexed · answers drawn only from this document
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
