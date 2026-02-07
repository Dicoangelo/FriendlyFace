import { useEffect, useState } from "react";

interface DashboardData {
  uptime_seconds: number;
  storage_backend: string;
  total_events: number;
  total_bundles: number;
  total_provenance_nodes: number;
  events_by_type: Record<string, number>;
  recent_events: Array<{
    id: string;
    event_type: string;
    actor: string;
    timestamp: string;
  }>;
  chain_integrity: { valid: boolean; count: number };
  crypto_status: {
    did_enabled: boolean;
    zk_scheme: string;
    total_dids: number;
    total_vcs: number;
  };
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState("");

  const fetchData = () => {
    fetch("/api/v1/dashboard")
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  };

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 10_000);
    return () => clearInterval(id);
  }, []);

  if (error) return <div className="text-red-600">Error loading dashboard: {error}</div>;

  if (!data)
    return (
      <div className="space-y-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-24 bg-gray-200 rounded animate-pulse" />
        ))}
      </div>
    );

  return (
    <div className="space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Events" value={data.total_events} />
        <StatCard label="Total Bundles" value={data.total_bundles} />
        <StatCard label="Provenance Nodes" value={data.total_provenance_nodes} />
        <StatCard
          label="Uptime"
          value={`${Math.floor(data.uptime_seconds / 60)}m ${Math.floor(data.uptime_seconds % 60)}s`}
        />
      </div>

      {/* Chain integrity + Crypto status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold text-gray-700 mb-3">Chain Integrity</h3>
          <div className="flex items-center gap-2">
            <span
              className={`w-3 h-3 rounded-full ${data.chain_integrity.valid ? "bg-green-500" : "bg-red-500"}`}
            />
            <span className={data.chain_integrity.valid ? "text-green-700" : "text-red-700"}>
              {data.chain_integrity.valid ? "Valid" : "Invalid"}
            </span>
            <span className="text-gray-400 text-sm">({data.chain_integrity.count} events)</span>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold text-gray-700 mb-3">Crypto Status</h3>
          <div className="text-sm space-y-1">
            <p>
              DID:{" "}
              <span className={data.crypto_status.did_enabled ? "text-green-600" : "text-gray-400"}>
                {data.crypto_status.did_enabled ? "Enabled" : "Disabled"}
              </span>
            </p>
            <p>ZK Scheme: {data.crypto_status.zk_scheme}</p>
            <p>DIDs: {data.crypto_status.total_dids} | VCs: {data.crypto_status.total_vcs}</p>
          </div>
        </div>
      </div>

      {/* Events by type */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="font-semibold text-gray-700 mb-3">Events by Type</h3>
        {Object.keys(data.events_by_type).length === 0 ? (
          <p className="text-gray-400 text-sm">No events recorded yet</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {Object.entries(data.events_by_type).map(([type, count]) => (
              <div key={type} className="bg-gray-50 rounded p-2">
                <p className="text-xs text-gray-500 truncate">{type}</p>
                <p className="text-lg font-semibold">{count}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent events */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="font-semibold text-gray-700 mb-3">Recent Events</h3>
        {data.recent_events.length === 0 ? (
          <p className="text-gray-400 text-sm">No events yet</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b">
                <th className="pb-2">Type</th>
                <th className="pb-2">Actor</th>
                <th className="pb-2">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {data.recent_events.map((e) => (
                <tr key={e.id} className="border-b last:border-0">
                  <td className="py-2">
                    <EventBadge type={e.event_type} />
                  </td>
                  <td className="py-2 text-gray-600">{e.actor}</td>
                  <td className="py-2 text-gray-400">{new Date(e.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="text-2xl font-bold text-gray-800 mt-1">{value}</p>
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
    consent_granted: "bg-teal-100 text-teal-700",
    consent_revoked: "bg-orange-100 text-orange-700",
  };
  const cls = colors[type] || "bg-gray-100 text-gray-700";
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>{type}</span>;
}
