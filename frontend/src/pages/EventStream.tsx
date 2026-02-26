import { useEffect, useRef, useState, useCallback } from "react";
import EmptyState from "../components/EmptyState";
import { eventBadgeColor, eventBorderColor } from "../constants/eventColors";

interface SSEEvent {
  id: string;
  event_type: string;
  actor: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

const MAX_RETRIES = 5;

export default function EventStream() {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [status, setStatus] = useState<"connecting" | "connected" | "reconnecting" | "disconnected">("connecting");
  const [paused, setPaused] = useState(false);
  const [filterType, setFilterType] = useState("");
  const [retryCount, setRetryCount] = useState(0);

  const esRef = useRef<EventSource | null>(null);
  const retryDelayRef = useRef(1000);
  const pausedRef = useRef(false);
  const listRef = useRef<HTMLDivElement>(null);
  const connectingRef = useRef(false);

  // Keep pausedRef in sync so the SSE handler closure always reads current value
  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  // Auto-scroll to bottom when new events arrive and not paused
  useEffect(() => {
    if (!paused && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [events, paused]);

  const connect = useCallback(() => {
    if (connectingRef.current) return;
    connectingRef.current = true;

    const url = filterType
      ? `/api/v1/events/stream?event_type=${filterType}`
      : "/api/v1/events/stream";
    const es = new EventSource(url);
    esRef.current = es;

    es.addEventListener("forensic_event", (e) => {
      if (pausedRef.current) return;
      try {
        const data = JSON.parse(e.data) as SSEEvent;
        setEvents((prev) => [...prev, data].slice(-100));
      } catch {
        // malformed event — skip
      }
    });

    es.addEventListener("heartbeat", () => {
      setStatus("connected");
    });

    es.onopen = () => {
      setStatus("connected");
      setRetryCount(0);
      retryDelayRef.current = 1000;
      connectingRef.current = false;
    };

    es.onerror = () => {
      es.close();
      connectingRef.current = false;

      setRetryCount((prev) => {
        const next = prev + 1;
        if (next >= MAX_RETRIES) {
          setStatus("disconnected");
          return next;
        }
        setStatus("reconnecting");
        const delay = Math.min(retryDelayRef.current, 30_000);
        retryDelayRef.current = delay * 2;
        setTimeout(connect, delay);
        return next;
      });
    };
  }, [filterType]);

  useEffect(() => {
    setStatus("connecting");
    setRetryCount(0);
    retryDelayRef.current = 1000;
    connect();
    return () => {
      esRef.current?.close();
      connectingRef.current = false;
    };
  }, [connect]);

  const handleRetry = () => {
    esRef.current?.close();
    connectingRef.current = false;
    setRetryCount(0);
    retryDelayRef.current = 1000;
    setStatus("connecting");
    connect();
  };

  const handleFilterChange = (value: string) => {
    esRef.current?.close();
    connectingRef.current = false;
    setFilterType(value);
  };

  const statusDot: Record<typeof status, string> = {
    connecting: "bg-gold animate-pulse",
    connected: "bg-teal shadow-lg shadow-teal/30",
    reconnecting: "bg-gold animate-pulse",
    disconnected: "bg-rose-ember shadow-lg shadow-rose-ember/30",
  };

  const statusLabel: Record<typeof status, string> = {
    connecting: "Connecting…",
    connected: "Connected",
    reconnecting: `Reconnecting (${retryCount}/${MAX_RETRIES})…`,
    disconnected: "Disconnected",
  };

  const statusPill: Record<typeof status, string> = {
    connecting: "bg-gold/10 text-gold border border-gold/20",
    connected: "bg-teal/10 text-teal border border-teal/20",
    reconnecting: "bg-gold/10 text-gold border border-gold/20",
    disconnected: "bg-rose-ember/10 text-rose-ember border border-rose-ember/20",
  };

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <span className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-medium ${statusPill[status]}`}>
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDot[status]}`} />
            {statusLabel[status]}
          </span>
          <span className="text-sm text-fg-muted">{events.length} events</span>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={filterType}
            onChange={(e) => handleFilterChange(e.target.value)}
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
            onClick={() => setPaused((p) => !p)}
            className={paused ? "btn-primary" : "btn-ghost"}
            title={paused ? "Resume auto-scroll" : "Pause auto-scroll"}
          >
            {paused ? "Resume scroll" : "Pause scroll"}
          </button>

          {events.length > 0 && (
            <button onClick={() => setEvents([])} className="btn-ghost text-fg-faint">
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Error banner — shown when max retries exceeded */}
      {status === "disconnected" && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-3 flex items-center justify-between">
          <div>
            <p className="text-rose-ember text-sm font-medium">SSE connection failed</p>
            <p className="text-rose-ember/70 text-xs mt-0.5">
              Could not connect after {MAX_RETRIES} attempts. Check that the server is running.
            </p>
          </div>
          <button onClick={handleRetry} className="btn-primary text-sm flex-shrink-0">
            Retry
          </button>
        </div>
      )}

      {/* Skeleton — shown while establishing first connection */}
      {status === "connecting" && events.length === 0 && (
        <div className="glass-card p-4 space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="flex items-center gap-3 py-2 border-b border-border-theme last:border-0">
              <div className="w-32 h-5 bg-surface rounded-full animate-pulse" />
              <div className="w-24 h-4 bg-surface rounded animate-pulse" />
              <div className="flex-1 h-4 bg-surface rounded animate-pulse" />
              <div className="w-20 h-4 bg-surface rounded animate-pulse" />
            </div>
          ))}
        </div>
      )}

      {/* Empty state — connected but no events yet */}
      {status === "connected" && events.length === 0 && (
        <div className="glass-card">
          <EmptyState
            title="Waiting for forensic events…"
            subtitle="Events will appear here in real-time as they are recorded via the API"
            icon={<div className="w-3 h-3 rounded-full bg-teal animate-pulse" />}
          />
        </div>
      )}

      {/* Event list — scrollable, newest at bottom */}
      {events.length > 0 && (
        <div
          ref={listRef}
          className="space-y-2 max-h-[calc(100vh-220px)] overflow-y-auto pr-1"
        >
          {events.map((evt, i) => (
            <EventRow key={`${evt.id}-${i}`} evt={evt} />
          ))}
          {/* Spacer so the last item isn't flush against the bottom */}
          <div className="h-2" />
        </div>
      )}
    </div>
  );
}

function EventRow({ evt }: { evt: SSEEvent }) {
  const [expanded, setExpanded] = useState(false);
  const badgeCls = eventBadgeColor(evt.event_type);
  const borderCls = eventBorderColor(evt.event_type);
  const shortId = evt.id ? evt.id.slice(0, 8) : "—";

  return (
    <div
      className={`glass-card p-3 border-l-2 ${borderCls} animate-[slideIn_0.3s_ease-out] cursor-pointer`}
      onClick={() => setExpanded((e) => !e)}
    >
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${badgeCls}`}>
            {evt.event_type}
          </span>
          <span className="text-xs text-fg-muted font-mono" title={evt.id}>
            {shortId}…
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-fg-secondary">{evt.actor}</span>
          <span className="text-xs text-fg-faint">
            {new Date(evt.timestamp).toLocaleTimeString()}
          </span>
        </div>
      </div>

      {expanded && (
        <pre className="text-xs text-fg-faint mt-2 overflow-x-auto max-h-32 font-mono bg-surface rounded p-2">
          {JSON.stringify(evt.payload, null, 2)}
        </pre>
      )}
    </div>
  );
}
