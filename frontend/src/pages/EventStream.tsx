import { useEffect, useRef, useState } from "react";

interface SSEEvent {
  id: string;
  event_type: string;
  actor: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

const EVENT_BADGE_COLORS: Record<string, string> = {
  training_start: "bg-amethyst/10 text-amethyst",
  training_complete: "bg-amethyst/10 text-amethyst",
  model_registered: "bg-amethyst/10 text-amethyst",
  inference_request: "bg-cyan/10 text-cyan",
  inference_result: "bg-cyan/10 text-cyan",
  explanation_generated: "bg-teal/10 text-teal",
  bias_audit: "bg-gold/10 text-gold",
  consent_recorded: "bg-teal/10 text-teal",
  consent_update: "bg-teal/10 text-teal",
  bundle_created: "bg-amethyst/10 text-amethyst",
  fl_round: "bg-cyan/10 text-cyan",
  security_alert: "bg-rose-ember/10 text-rose-ember",
  compliance_report: "bg-gold/10 text-gold",
};

const EVENT_BORDER_COLORS: Record<string, string> = {
  training_start: "border-amethyst/30",
  training_complete: "border-amethyst/30",
  model_registered: "border-amethyst/30",
  inference_request: "border-cyan/30",
  inference_result: "border-cyan/30",
  explanation_generated: "border-teal/30",
  bias_audit: "border-gold/30",
  consent_recorded: "border-teal/30",
  consent_update: "border-teal/30",
  bundle_created: "border-amethyst/30",
  fl_round: "border-cyan/30",
  security_alert: "border-rose-ember/30",
  compliance_report: "border-gold/30",
};

export default function EventStream() {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [status, setStatus] = useState<"connected" | "reconnecting" | "disconnected">(
    "disconnected",
  );
  const [paused, setPaused] = useState(false);
  const [filterType, setFilterType] = useState("");
  const esRef = useRef<EventSource | null>(null);
  const retryRef = useRef(1000);

  const connect = () => {
    const url = filterType ? `/api/v1/events/stream?event_type=${filterType}` : "/api/v1/events/stream";
    const es = new EventSource(url);
    esRef.current = es;

    es.addEventListener("forensic_event", (e) => {
      const data = JSON.parse(e.data) as SSEEvent;
      setEvents((prev) => [data, ...prev].slice(0, 100));
    });

    es.addEventListener("heartbeat", () => {
      setStatus("connected");
    });

    es.onopen = () => {
      setStatus("connected");
      retryRef.current = 1000;
    };

    es.onerror = () => {
      es.close();
      setStatus("reconnecting");
      const delay = Math.min(retryRef.current, 30000);
      retryRef.current = delay * 2;
      setTimeout(connect, delay);
    };
  };

  useEffect(() => {
    connect();
    return () => {
      esRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterType]);

  const statusColors = {
    connected: "bg-teal/10 text-teal border border-teal/20",
    reconnecting: "bg-gold/10 text-gold border border-gold/20",
    disconnected: "bg-rose-ember/10 text-rose-ember border border-rose-ember/20",
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-bold text-fg">Live Event Stream</h2>
          <span className={`px-2 py-1 rounded-lg text-xs font-medium ${statusColors[status]}`}>
            {status}
          </span>
          <span className="text-sm text-fg-muted">{events.length} events</span>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={filterType}
            onChange={(e) => {
              esRef.current?.close();
              setFilterType(e.target.value);
            }}
            className="ff-select"
          >
            <option value="">All types</option>
            <option value="training_start">training_start</option>
            <option value="training_complete">training_complete</option>
            <option value="model_registered">model_registered</option>
            <option value="inference_request">inference_request</option>
            <option value="inference_result">inference_result</option>
            <option value="explanation_generated">explanation_generated</option>
            <option value="bias_audit">bias_audit</option>
            <option value="consent_recorded">consent_recorded</option>
            <option value="consent_update">consent_update</option>
            <option value="bundle_created">bundle_created</option>
            <option value="fl_round">fl_round</option>
            <option value="security_alert">security_alert</option>
            <option value="compliance_report">compliance_report</option>
          </select>
          <button
            onClick={() => setPaused(!paused)}
            className={paused ? "btn-primary" : "btn-ghost"}
          >
            {paused ? "Resume" : "Pause"}
          </button>
          {events.length > 0 && (
            <button
              onClick={() => setEvents([])}
              className="btn-ghost text-fg-faint"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {events.length === 0 ? (
        <div className="glass-card p-12 text-center">
          <p className="text-fg-faint text-lg mb-1">Waiting for events...</p>
          <p className="text-fg-faint text-sm">Events will appear here in real-time as they are recorded via the API.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {events.map((evt, i) => {
            const badgeCls = EVENT_BADGE_COLORS[evt.event_type] || "bg-fg/5 text-fg-secondary";
            const borderCls = EVENT_BORDER_COLORS[evt.event_type] || "border-fg-faint/30";
            return (
              <div
                key={`${evt.id}-${i}`}
                className={`glass-card p-3 border-l-2 ${borderCls} animate-[slideIn_0.3s_ease-out]`}
              >
                <div className="flex items-center justify-between">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${badgeCls}`}>
                    {evt.event_type}
                  </span>
                  <span className="text-xs text-fg-faint">
                    {new Date(evt.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-sm text-fg-secondary mt-1">Actor: {evt.actor}</p>
                <pre className="text-xs text-fg-faint mt-1 overflow-x-auto max-h-20 font-mono">
                  {JSON.stringify(evt.payload, null, 2)}
                </pre>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
