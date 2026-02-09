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

interface FLRound {
  simulation_id: string;
  round_number: number;
  global_accuracy: number;
  num_clients: number;
  dp_enabled: boolean;
  timestamp: string;
}

interface RecognitionModel {
  model_id: string;
  model_type: string;
  accuracy: number;
  trained_at: string;
  num_classes: number;
}

const EVENT_TYPE_BAR_COLORS: Record<string, { bg: string; text: string }> = {
  training_start: { bg: "bg-amethyst/30", text: "text-amethyst" },
  training_complete: { bg: "bg-amethyst/30", text: "text-amethyst" },
  model_registered: { bg: "bg-amethyst/30", text: "text-amethyst" },
  inference_request: { bg: "bg-cyan/30", text: "text-cyan" },
  inference_result: { bg: "bg-cyan/30", text: "text-cyan" },
  explanation_generated: { bg: "bg-teal/30", text: "text-teal" },
  bias_audit: { bg: "bg-gold/30", text: "text-gold" },
  consent_recorded: { bg: "bg-teal/30", text: "text-teal" },
  consent_update: { bg: "bg-teal/30", text: "text-teal" },
  bundle_created: { bg: "bg-amethyst/30", text: "text-amethyst" },
  fl_round: { bg: "bg-cyan/30", text: "text-cyan" },
  security_alert: { bg: "bg-rose-ember/30", text: "text-rose-ember" },
  compliance_report: { bg: "bg-gold/30", text: "text-gold" },
};

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [flRounds, setFlRounds] = useState<FLRound[]>([]);
  const [models, setModels] = useState<RecognitionModel[]>([]);
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

    fetch("/api/v1/fl/rounds")
      .then((r) => r.json())
      .then((data) => setFlRounds(Array.isArray(data) ? data : []))
      .catch(() => {});

    fetch("/api/v1/recognition/models")
      .then((r) => r.json())
      .then((data) => setModels(Array.isArray(data) ? data : []))
      .catch(() => {});

    return () => clearInterval(id);
  }, []);

  if (error) return <div className="text-rose-ember">Error loading dashboard: {error}</div>;

  if (!data)
    return (
      <div className="space-y-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-24 bg-surface rounded-lg animate-pulse" />
        ))}
      </div>
    );

  return (
    <div className="space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Events" value={data.total_events} color="cyan" />
        <StatCard label="Total Bundles" value={data.total_bundles} color="amethyst" />
        <StatCard label="Provenance Nodes" value={data.total_provenance_nodes} color="teal" />
        <StatCard
          label="Uptime"
          value={`${Math.floor(data.uptime_seconds / 60)}m ${Math.floor(data.uptime_seconds % 60)}s`}
          color="gold"
        />
      </div>

      {/* Chain integrity + Crypto status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-3">Chain Integrity</h3>
          <div className="flex items-center gap-2">
            <span
              className={`w-3 h-3 rounded-full ${data.chain_integrity.valid ? "bg-teal shadow-lg shadow-teal/30" : "bg-rose-ember shadow-lg shadow-rose-ember/30"}`}
            />
            <span className={data.chain_integrity.valid ? "text-teal" : "text-rose-ember"}>
              {data.chain_integrity.valid ? "Valid" : "Invalid"}
            </span>
            <span className="text-fg-faint text-sm">({data.chain_integrity.count} events)</span>
          </div>
        </div>
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-3">Crypto Status</h3>
          <div className="text-sm space-y-1">
            <p>
              DID:{" "}
              <span className={data.crypto_status.did_enabled ? "text-teal" : "text-fg-faint"}>
                {data.crypto_status.did_enabled ? "Enabled" : "Disabled"}
              </span>
            </p>
            <p className="text-fg-secondary">ZK Scheme: {data.crypto_status.zk_scheme}</p>
            <p className="text-fg-secondary">DIDs: {data.crypto_status.total_dids} | VCs: {data.crypto_status.total_vcs}</p>
          </div>
        </div>
      </div>

      {/* Events by type */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Events by Type</h3>
        {Object.keys(data.events_by_type).length === 0 ? (
          <p className="text-fg-faint text-sm">No events recorded yet</p>
        ) : (
          <div className="space-y-2">
            {Object.entries(data.events_by_type)
              .sort(([, a], [, b]) => b - a)
              .map(([type, count]) => {
                const maxCount = Math.max(...Object.values(data.events_by_type));
                const pct = maxCount > 0 ? (count / maxCount) * 100 : 0;
                const color = EVENT_TYPE_BAR_COLORS[type] || { bg: "bg-fg/10", text: "text-fg-secondary" };
                return (
                  <div key={type} className="flex items-center gap-3">
                    <span className="text-xs text-fg-muted w-40 truncate">{type.replace(/_/g, " ")}</span>
                    <div className="flex-1 h-6 bg-surface rounded-md overflow-hidden relative">
                      <div
                        className={`h-full rounded-md ${color.bg} transition-all duration-500`}
                        style={{ width: `${pct}%` }}
                      />
                      <span className={`absolute inset-y-0 right-2 flex items-center text-xs font-semibold ${color.text}`}>
                        {count}
                      </span>
                    </div>
                  </div>
                );
              })}
          </div>
        )}
      </div>

      {/* FL Simulation History + Model Registry row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* FL Simulation History */}
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-3">FL Simulation History</h3>
          {flRounds.length === 0 ? (
            <p className="text-fg-faint text-sm">No FL rounds completed yet</p>
          ) : (
            <div className="space-y-2">
              {flRounds.slice(0, 5).map((r, i) => (
                <div key={i} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                  <div>
                    <p className="text-sm text-fg-secondary">
                      {r.simulation_id.slice(0, 8)} — Round {r.round_number}
                    </p>
                    <p className="text-xs text-fg-faint">
                      {r.num_clients} client{r.num_clients !== 1 ? "s" : ""}{r.dp_enabled ? " + DP" : ""}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className={`text-sm font-semibold ${r.global_accuracy >= 0.8 ? "text-teal" : r.global_accuracy >= 0.5 ? "text-gold" : "text-rose-ember"}`}>
                      {(r.global_accuracy * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-fg-faint">accuracy</p>
                  </div>
                </div>
              ))}
              {flRounds.length > 5 && (
                <p className="text-xs text-fg-faint text-center">+{flRounds.length - 5} more rounds</p>
              )}
            </div>
          )}
        </div>

        {/* Model Registry */}
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-3">Recognition Models</h3>
          {models.length === 0 ? (
            <p className="text-fg-faint text-sm">No models trained yet</p>
          ) : (
            <div className="space-y-2">
              {models.slice(0, 5).map((m) => (
                <div key={m.model_id} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                  <div>
                    <p className="text-sm text-fg-secondary font-mono">{m.model_id.slice(0, 12)}</p>
                    <p className="text-xs text-fg-faint">
                      {m.model_type} — {m.num_classes} classes
                    </p>
                  </div>
                  <div className="text-right">
                    <p className={`text-sm font-semibold ${m.accuracy >= 0.9 ? "text-teal" : m.accuracy >= 0.7 ? "text-gold" : "text-rose-ember"}`}>
                      {(m.accuracy * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-fg-faint">
                      {new Date(m.trained_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Recent events */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Recent Events</h3>
        {data.recent_events.length === 0 ? (
          <p className="text-fg-faint text-sm">No events yet</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-fg-muted border-b border-border-theme">
                <th className="pb-2">Type</th>
                <th className="pb-2">Actor</th>
                <th className="pb-2">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {data.recent_events.map((e) => (
                <tr key={e.id} className="border-b border-border-theme last:border-0">
                  <td className="py-2">
                    <EventBadge type={e.event_type} />
                  </td>
                  <td className="py-2 text-fg-secondary">{e.actor}</td>
                  <td className="py-2 text-fg-faint">{new Date(e.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

const STAT_COLORS: Record<string, string> = {
  cyan: "border-cyan/20 text-cyan",
  amethyst: "border-amethyst/20 text-amethyst",
  teal: "border-teal/20 text-teal",
  gold: "border-gold/20 text-gold",
};

function StatCard({ label, value, color = "cyan" }: { label: string; value: string | number; color?: string }) {
  return (
    <div className={`glass-card p-4 border-l-2 ${STAT_COLORS[color] || STAT_COLORS.cyan}`}>
      <p className="text-sm text-fg-muted">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${STAT_COLORS[color]?.split(" ")[1] || "text-cyan"}`}>{value}</p>
    </div>
  );
}

function EventBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
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
  const cls = colors[type] || "bg-fg/5 text-fg-secondary";
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>{type}</span>;
}
