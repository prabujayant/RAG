import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-[3px] text-[10px] font-semibold uppercase tracking-[0.12em] leading-none",
  {
    variants: {
      variant: {
        default: "border-[#111111]/20 bg-[#111111] text-white",
        secondary: "border-[#111111]/10 bg-[#f4f4f5] text-[#4a4a4a]",
        success: "border-[#1a7a4a]/20 bg-[#d6efe4] text-[#15683e]",
        warning: "border-[#9a6400]/20 bg-[#fef3c7] text-[#8a5a00]",
        danger: "border-[#d62839]/20 bg-[#fdeaea] text-[#b5202f]",
        outline: "border-[#111111]/20 bg-white/60 text-[#111111]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
