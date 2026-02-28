import { useState } from "react";

interface DemoBannerProps {
    visible: boolean;
}

export default function DemoBanner({ visible }: DemoBannerProps) {
    const [dismissed, setDismissed] = useState(false);

    if (!visible || dismissed) return null;

    return (
        <div className="relative bg-gradient-to-r from-amethyst/15 via-cyan/15 to-amethyst/15 border-b border-cyan/20 px-4 py-2.5 text-center text-sm">
            <span className="text-fg-secondary">
                <span className="inline-block mr-1.5 animate-pulse">🔬</span>
                <strong className="text-fg font-semibold">Demo Mode</strong>
                {" — "}
                Showing sample data. Connect a backend at{" "}
                <code className="px-1.5 py-0.5 rounded text-xs bg-surface font-mono text-cyan">
                    localhost:3849
                </code>
                {" "}for live data.
            </span>
            <button
                onClick={() => setDismissed(true)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-fg-faint hover:text-fg transition-colors p-1"
                aria-label="Dismiss demo banner"
            >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>
    );
}
