import { createContext } from "react";

type Theme = "light" | "dark";

export interface ThemeContextValue {
  theme: Theme;
  toggle: () => void;
}

export const ThemeContext = createContext<ThemeContextValue>({ theme: "dark", toggle: () => {} });
