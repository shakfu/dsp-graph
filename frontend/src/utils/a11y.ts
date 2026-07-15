import type { KeyboardEvent, SyntheticEvent } from "react";

/**
 * Props that make a non-native element behave like a button for both mouse and
 * keyboard: it is focusable, has the button role, and activates on Enter/Space.
 * Use for intentionally-custom clickable <div>/<span> elements (menu items,
 * list rows) so they are keyboard-accessible.
 */
export function buttonProps(onActivate: () => void): {
  role: "button";
  tabIndex: 0;
  onClick: () => void;
  onKeyDown: (e: KeyboardEvent) => void;
} {
  return {
    role: "button",
    tabIndex: 0,
    onClick: onActivate,
    onKeyDown: (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onActivate();
      }
    },
  };
}

/**
 * Background-swap highlight handlers that respond to both pointer hover and
 * keyboard focus (so the highlight is visible when tabbing, not only on hover).
 */
export function hoverHighlight(bg = "#f0f0f0"): {
  onMouseOver: (e: SyntheticEvent<HTMLElement>) => void;
  onMouseOut: (e: SyntheticEvent<HTMLElement>) => void;
  onFocus: (e: SyntheticEvent<HTMLElement>) => void;
  onBlur: (e: SyntheticEvent<HTMLElement>) => void;
} {
  const set = (color: string) => (e: SyntheticEvent<HTMLElement>) => {
    e.currentTarget.style.background = color;
  };
  const on = set(bg);
  const off = set("transparent");
  return { onMouseOver: on, onMouseOut: off, onFocus: on, onBlur: off };
}
