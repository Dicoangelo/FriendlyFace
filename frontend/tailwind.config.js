/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        midnight: "#0B1020",
        obsidian: "#05070D",
        surface: "rgb(var(--color-surface) / <alpha-value>)",
        "surface-light": "rgb(var(--color-surface-light) / <alpha-value>)",
        cyan: { DEFAULT: "#18E6FF", dim: "#0EA5BB" },
        amethyst: { DEFAULT: "#7B2CFF", dim: "#5B1ED4" },
        magenta: "#FF3DF2",
        "holo-blue": "#5AA7FF",
        teal: "#00FFC6",
        "rose-ember": "#FF6B8A",
        gold: "#D7B26D",
        // Semantic tokens via CSS vars
        fg: "rgb(var(--color-fg) / <alpha-value>)",
        "fg-secondary": "rgb(var(--color-fg-secondary) / <alpha-value>)",
        "fg-muted": "rgb(var(--color-fg-muted) / <alpha-value>)",
        "fg-faint": "rgb(var(--color-fg-faint) / <alpha-value>)",
        page: "rgb(var(--color-page) / <alpha-value>)",
        card: "rgb(var(--color-card) / <alpha-value>)",
        sidebar: "rgb(var(--color-sidebar) / <alpha-value>)",
        "border-theme": "rgb(var(--color-border) / <alpha-value>)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      keyframes: {
        "slide-in": {
          "0%": { transform: "translateX(100%)", opacity: "0" },
          "100%": { transform: "translateX(0)", opacity: "1" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "slide-in": "slide-in 0.3s ease-out",
        shimmer: "shimmer 2s linear infinite",
      },
    },
  },
  plugins: [],
};
