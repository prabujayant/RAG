import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[6px] text-[11px] font-semibold uppercase tracking-[0.14em] transition-[background-color,box-shadow,transform,color] duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d62839]/50 focus-visible:ring-offset-2 focus-visible:ring-offset-white disabled:pointer-events-none disabled:opacity-45 active:translate-y-px [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-[#d62839] text-white shadow-[0_1px_2px_rgba(168,31,45,0.35),inset_0_1px_0_rgba(255,255,255,0.18)] hover:bg-[#bf2130] hover:shadow-[0_4px_14px_-4px_rgba(214,40,57,0.55)]",
        secondary:
          "bg-[#111111] text-white shadow-[0_1px_2px_rgba(17,17,17,0.3),inset_0_1px_0_rgba(255,255,255,0.12)] hover:bg-[#2a2a2a]",
        outline:
          "border border-[#111111]/20 bg-white text-[#111111] hover:border-[#111111]/35 hover:bg-[#f6f6f6]",
        ghost: "bg-transparent text-[#111111] hover:bg-[#111111]/6",
      },
      size: {
        default: "h-11 px-5 py-2",
        sm: "h-9 px-3",
        lg: "h-12 px-6",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
