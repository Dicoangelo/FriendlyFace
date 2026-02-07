import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:3849",
      "/health": "http://localhost:3849",
      "/events": "http://localhost:3849",
      "/dashboard": "http://localhost:3849",
      "/bundles": "http://localhost:3849",
      "/did": "http://localhost:3849",
      "/vc": "http://localhost:3849",
      "/zk": "http://localhost:3849",
      "/fl": "http://localhost:3849",
      "/fairness": "http://localhost:3849",
      "/consent": "http://localhost:3849",
      "/governance": "http://localhost:3849",
      "/recognition": "http://localhost:3849",
      "/explainability": "http://localhost:3849",
      "/merkle": "http://localhost:3849",
      "/chain": "http://localhost:3849",
      "/provenance": "http://localhost:3849",
      "/verify": "http://localhost:3849",
      "/recognize": "http://localhost:3849",
    },
  },
  build: {
    outDir: "dist",
  },
});
