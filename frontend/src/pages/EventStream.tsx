import { useEffect, useRef, useState } from "react";
import EmptyState from "../components/EmptyState";
import { eventBadgeColor, eventBorderColor } from "../constants/eventColors";

interface SSEEvent {
  id: string;
  event_type: string;
  actor: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

// Color constants imported from shared eventColors

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
          {/* Page title shown in header bar */}
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
        <div className="glass-card">
          <EmptyState
            title="Waiting for events..."
            subtitle="Events will appear here in real-time as they are recorded via the API"
            icon={<div className="w-3 h-3 rounded-full bg-teal animate-pulse" />}
          />
        </div>
      ) : (
        <div className="space-y-2">
          {events.map((evt, i) => {
            const badgeCls = eventBadgeColor(evt.event_type);
            const borderCls = eventBorderColor(evt.event_type);
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
