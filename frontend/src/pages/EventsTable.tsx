import { Fragment, useEffect, useState } from "react";
import { SkeletonTable } from "../components/Skeleton";
import { useCopyToClipboard } from "../hooks/useCopyToClipboard";

interface ForensicEvent {
  id: string;
  event_type: string;
  actor: string;
  timestamp: string;
  sequence_number: number;
  payload: Record<string, unknown>;
}

const EVENT_TYPE_COLORS: Record<string, string> = {
  training_start: "bg-amethyst/10 text-amethyst border-amethyst/20",
  training_complete: "bg-amethyst/10 text-amethyst border-amethyst/20",
  model_registered: "bg-amethyst/10 text-amethyst border-amethyst/20",
  inference_request: "bg-cyan/10 text-cyan border-cyan/20",
  inference_result: "bg-cyan/10 text-cyan border-cyan/20",
  explanation_generated: "bg-teal/10 text-teal border-teal/20",
  bias_audit: "bg-gold/10 text-gold border-gold/20",
  consent_recorded: "bg-teal/10 text-teal border-teal/20",
  consent_update: "bg-teal/10 text-teal border-teal/20",
  bundle_created: "bg-amethyst/10 text-amethyst border-amethyst/20",
  fl_round: "bg-cyan/10 text-cyan border-cyan/20",
  security_alert: "bg-rose-ember/10 text-rose-ember border-rose-ember/20",
  compliance_report: "bg-gold/10 text-gold border-gold/20",
};

export default function EventsTable() {
  const copy = useCopyToClipboard();
  const [events, setEvents] = useState<ForensicEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [activeType, setActiveType] = useState<string | null>(null);
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

  // Count events by type
  const typeCounts: Record<string, number> = {};
  for (const e of events) {
    typeCounts[e.event_type] = (typeCounts[e.event_type] || 0) + 1;
  }
  const sortedTypes = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]);

  const filtered = events.filter((e) => {
    const matchesType = !activeType || e.event_type === activeType;
    const matchesSearch =
      !search ||
      e.event_type.includes(search) ||
      e.actor.includes(search) ||
      e.id.includes(search);
    return matchesType && matchesSearch;
  });

  if (loading) return <SkeletonTable rows={8} />;

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

      {/* Type filter chips */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setActiveType(null)}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
            activeType === null
              ? "bg-cyan/10 text-cyan border-cyan/30"
              : "bg-surface text-fg-muted border-border-theme hover:border-fg-faint/30"
          }`}
        >
          All ({events.length})
        </button>
        {sortedTypes.map(([type, count]) => {
          const colors = EVENT_TYPE_COLORS[type] || "bg-fg/5 text-fg-secondary border-fg-faint/20";
          const isActive = activeType === type;
          return (
            <button
              key={type}
              onClick={() => setActiveType(isActive ? null : type)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                isActive ? colors : "bg-surface text-fg-muted border-border-theme hover:border-fg-faint/30"
              }`}
            >
              {type.replace(/_/g, " ")} ({count})
            </button>
          );
        })}
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
              <Fragment key={e.id}>
                <tr
                  onClick={() => setExpandedId(expandedId === e.id ? null : e.id)}
                  className="border-b border-border-theme hover:bg-fg/[0.02] cursor-pointer"
                >
                  <td className="px-4 py-2 font-mono text-xs text-fg-secondary">
                    {e.id.slice(0, 8)}...
                    <button
                      onClick={(ev) => {
                        ev.stopPropagation();
                        copy(e.id, "Event ID copied");
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
                  <tr>
                    <td colSpan={5} className="px-4 py-3 bg-surface">
                      <pre className="text-xs overflow-x-auto whitespace-pre-wrap text-fg-muted font-mono">
                        {JSON.stringify(e.payload, null, 2)}
                      </pre>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <p className="text-center py-8 text-fg-faint">No events match your filters</p>
        )}
      </div>
    </div>
  );
}

function EventBadge({ type }: { type: string }) {
  const colors = EVENT_TYPE_COLORS[type] || "bg-fg/5 text-fg-secondary border-transparent";
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors.split(" ").slice(0, 2).join(" ")}`}>{type}</span>;
}
