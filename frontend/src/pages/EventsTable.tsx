import { useEffect, useState } from "react";

interface ForensicEvent {
  id: string;
  event_type: string;
  actor: string;
  timestamp: string;
  sequence_number: number;
  payload: Record<string, unknown>;
}

export default function EventsTable() {
  const [events, setEvents] = useState<ForensicEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/v1/events")
      .then((r) => r.json())
      .then((data) => {
        setEvents(data.items || data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const filtered = events.filter(
    (e) =>
      e.event_type.includes(search) ||
      e.actor.includes(search) ||
      e.id.includes(search),
  );

  if (loading)
    return <div className="h-48 bg-surface rounded-lg animate-pulse" />;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-fg">Forensic Events ({events.length})</h2>
        <input
          type="text"
          placeholder="Search by type, actor, or ID..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="ff-input w-64"
        />
      </div>

      <div className="glass-card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-surface">
            <tr className="text-left text-fg-muted">
              <th className="px-4 py-3">ID</th>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3">Actor</th>
              <th className="px-4 py-3">Seq #</th>
              <th className="px-4 py-3">Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((e) => (
              <>
                <tr
                  key={e.id}
                  onClick={() => setExpandedId(expandedId === e.id ? null : e.id)}
                  className="border-b border-border-theme hover:bg-fg/[0.02] cursor-pointer"
                >
                  <td className="px-4 py-2 font-mono text-xs text-fg-secondary">
                    {e.id.slice(0, 8)}...
                    <button
                      onClick={(ev) => {
                        ev.stopPropagation();
                        navigator.clipboard.writeText(e.id);
                      }}
                      className="ml-1 text-fg-faint hover:text-cyan"
                      title="Copy ID"
                    >
                      <svg className="w-3 h-3 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                    </button>
                  </td>
                  <td className="px-4 py-2">
                    <EventBadge type={e.event_type} />
                  </td>
                  <td className="px-4 py-2 text-fg-secondary">{e.actor}</td>
                  <td className="px-4 py-2 text-fg-muted">{e.sequence_number}</td>
                  <td className="px-4 py-2 text-fg-faint">
                    {new Date(e.timestamp).toLocaleString()}
                  </td>
                </tr>
                {expandedId === e.id && (
                  <tr key={`${e.id}-detail`}>
                    <td colSpan={5} className="px-4 py-3 bg-surface">
                      <pre className="text-xs overflow-x-auto whitespace-pre-wrap text-fg-muted font-mono">
                        {JSON.stringify(e.payload, null, 2)}
                      </pre>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <p className="text-center py-8 text-fg-faint">No events match your search</p>
        )}
      </div>
    </div>
  );
}

function EventBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    inference_result: "bg-cyan/10 text-cyan",
    training_complete: "bg-amethyst/10 text-amethyst",
    security_alert: "bg-rose-ember/10 text-rose-ember",
    bias_audit_complete: "bg-gold/10 text-gold",
    explanation_generated: "bg-teal/10 text-teal",
  };
  const cls = colors[type] || "bg-fg/5 text-fg-secondary";
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>{type}</span>;
}
