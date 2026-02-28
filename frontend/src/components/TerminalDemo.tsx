import { useEffect, useState } from "react";

const COMMANDS = [
    {
        prompt: "curl -s http://localhost:3849/health | jq .",
        response: `{
  "status": "healthy",
  "version": "0.1.0",
  "storage": "sqlite",
  "uptime_seconds": 86400
}`,
    },
    {
        prompt: 'curl -s -X POST http://localhost:3849/events \\\n  -H "Content-Type: application/json" \\\n  -d \'{"event_type":"recognition_inference","actor":"system","payload":{"confidence":0.96}}\' | jq .id',
        response: `"a4f7c8e2-1d3b-4a5c-b7d9-e0f2a3b4c5d6"`,
    },
    {
        prompt: "curl -s http://localhost:3849/merkle/root | jq .",
        response: `{
  "root": "sha256:4a7d1ed4...c6d7e8",
  "leaf_count": 1248,
  "tree_height": 11
}`,
    },
    {
        prompt: "curl -s http://localhost:3849/fairness/status | jq .",
        response: `{
  "demographic_parity_gap": 0.04,
  "equalized_odds_gap": 0.06,
  "compliant": true,
  "groups_audited": 4
}`,
    },
];

export default function TerminalDemo() {
    const [currentCmd, setCurrentCmd] = useState(0);
    const [typedChars, setTypedChars] = useState(0);
    const [showResponse, setShowResponse] = useState(false);
    const [phase, setPhase] = useState<"typing" | "response" | "pause">("typing");

    const command = COMMANDS[currentCmd];
    const totalChars = command.prompt.length;

    useEffect(() => {
        let timer: ReturnType<typeof setTimeout>;

        if (phase === "typing") {
            if (typedChars < totalChars) {
                timer = setTimeout(() => setTypedChars((c: number) => c + 1), 25);
            } else {
                timer = setTimeout(() => {
                    setShowResponse(true);
                    setPhase("response");
                }, 300);
            }
        } else if (phase === "response") {
            timer = setTimeout(() => setPhase("pause"), 2500);
        } else if (phase === "pause") {
            timer = setTimeout(() => {
                setTypedChars(0);
                setShowResponse(false);
                setCurrentCmd((c: number) => (c + 1) % COMMANDS.length);
                setPhase("typing");
            }, 500);
        }

        return () => clearTimeout(timer);
    }, [phase, typedChars, totalChars]);

    return (
        <div className="glass-card overflow-hidden max-w-2xl mx-auto">
            {/* Terminal title bar */}
            <div className="flex items-center gap-2 px-4 py-2.5 bg-surface border-b border-border-theme">
                <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-rose-ember/70" />
                    <div className="w-3 h-3 rounded-full bg-gold/70" />
                    <div className="w-3 h-3 rounded-full bg-teal/70" />
                </div>
                <span className="text-xs text-fg-faint font-mono ml-2">friendlyface — bash</span>
            </div>

            {/* Terminal content */}
            <div className="p-4 font-mono text-sm leading-relaxed" style={{ background: "rgb(var(--terminal-bg))" }}>
                <div className="text-fg-muted">
                    <span className="text-teal">$</span>{" "}
                    <span className="text-fg-secondary">
                        {command.prompt.slice(0, typedChars)}
                    </span>
                    {phase === "typing" && (
                        <span className="inline-block w-2 h-4 bg-cyan animate-pulse ml-0.5 align-text-bottom" />
                    )}
                </div>

                {showResponse && (
                    <pre className="mt-2 text-cyan/90 whitespace-pre-wrap animate-fade-in text-xs">
                        {command.response}
                    </pre>
                )}
            </div>
        </div>
    );
}
