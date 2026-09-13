import * as React from "react";

import { cn } from "@/lib/utils";

const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[96px] w-full rounded-[6px] border border-[#111111]/15 bg-white px-3.5 py-3 text-sm text-[#111111] shadow-[inset_0_1px_2px_rgba(17,17,17,0.04)] transition-[border-color,box-shadow] placeholder:text-[#9a9a9a] hover:border-[#111111]/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d62839]/40 focus-visible:border-[#d62839] disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Textarea.displayName = "Textarea";

export { Textarea };
