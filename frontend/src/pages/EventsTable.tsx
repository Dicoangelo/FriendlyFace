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
    return <div className="h-48 bg-gray-200 rounded animate-pulse" />;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Forensic Events ({events.length})</h2>
        <input
          type="text"
          placeholder="Search by type, actor, or ID..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="border rounded px-3 py-1 text-sm w-64"
        />
      </div>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr className="text-left text-gray-500">
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
                  className="border-b hover:bg-gray-50 cursor-pointer"
                >
                  <td className="px-4 py-2 font-mono text-xs">
                    {e.id.slice(0, 8)}...
                    <button
                      onClick={(ev) => {
                        ev.stopPropagation();
                        navigator.clipboard.writeText(e.id);
                      }}
                      className="ml-1 text-gray-400 hover:text-gray-600"
                      title="Copy ID"
                    >
                      📋
                    </button>
                  </td>
                  <td className="px-4 py-2">
                    <EventBadge type={e.event_type} />
                  </td>
                  <td className="px-4 py-2 text-gray-600">{e.actor}</td>
                  <td className="px-4 py-2 text-gray-500">{e.sequence_number}</td>
                  <td className="px-4 py-2 text-gray-400">
                    {new Date(e.timestamp).toLocaleString()}
                  </td>
                </tr>
                {expandedId === e.id && (
                  <tr key={`${e.id}-detail`}>
                    <td colSpan={5} className="px-4 py-3 bg-gray-50">
                      <pre className="text-xs overflow-x-auto whitespace-pre-wrap">
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
          <p className="text-center py-8 text-gray-400">No events match your search</p>
        )}
      </div>
    </div>
  );
}

function EventBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    inference_result: "bg-blue-100 text-blue-700",
    training_complete: "bg-purple-100 text-purple-700",
    security_alert: "bg-red-100 text-red-700",
    bias_audit_complete: "bg-yellow-100 text-yellow-700",
    explanation_generated: "bg-green-100 text-green-700",
  };
  const cls = colors[type] || "bg-gray-100 text-gray-700";
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>{type}</span>;
}
