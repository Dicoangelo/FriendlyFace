import { useEffect, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import { SkeletonDashboard } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";
import ProgressRing from "../components/ProgressRing";
import { eventBarColor, eventBadgeColor } from "../constants/eventColors";

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

// Bar chart color constants imported from shared eventColors

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [flRounds, setFlRounds] = useState<FLRound[]>([]);
  const [models, setModels] = useState<RecognitionModel[]>([]);
  const [galleryCount, setGalleryCount] = useState<number | null>(null);
  const [compliance, setCompliance] = useState<{ compliant: boolean; overall_score: number; requirements?: Array<{ name: string; status: string }> } | null>(null);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [secondsAgo, setSecondsAgo] = useState(0);

  const fetchData = useCallback(() => {
    fetch("/api/v1/dashboard")
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((d) => {
        setData(d);
        setLastUpdated(new Date());
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    const tick = setInterval(() => {
      if (lastUpdated) setSecondsAgo(Math.floor((Date.now() - lastUpdated.getTime()) / 1000));
    }, 1000);
    return () => clearInterval(tick);
  }, [lastUpdated]);

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

    fetch("/api/v1/gallery/count")
      .then((r) => r.json())
      .then((data) => setGalleryCount(data.total ?? data.count ?? 0))
      .catch(() => {});

    fetch("/api/v1/governance/compliance")
      .then((r) => r.json())
      .then(setCompliance)
      .catch(() => {});

    return () => clearInterval(id);
  }, [fetchData]);

  if (error) return <div className="text-rose-ember">Error loading dashboard: {error}</div>;

  if (!data) return <SkeletonDashboard />;

  const formatUptime = (s: number) => {
    if (s >= 3600) {
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      const sec = Math.floor(s % 60);
      return `${h}h ${m}m ${sec}s`;
    }
    return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  };

  return (
    <div className="space-y-6">
      {/* Refresh bar */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-fg-faint">
          {lastUpdated ? `Last updated ${secondsAgo}s ago` : "Loading..."}
        </span>
        <button onClick={fetchData} className="btn-ghost text-sm inline-flex items-center gap-1.5">
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
          Refresh
        </button>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard label="Total Events" value={data.total_events} color="cyan" />
        <StatCard label="Total Bundles" value={data.total_bundles} color="amethyst" />
        <StatCard label="Provenance Nodes" value={data.total_provenance_nodes} color="teal" />
        <StatCard
          label="Uptime"
          value={formatUptime(data.uptime_seconds)}
          color="gold"
        />
        {galleryCount !== null && (
          <StatCard label="Gallery Subjects" value={galleryCount} color="cyan" />
        )}
        {compliance && (
          <div className={`glass-card p-4 border-l-2 ${compliance.compliant ? "border-teal/20" : "border-rose-ember/20"}`}>
            <p className="text-sm text-fg-muted">Compliance</p>
            <div className="flex items-center gap-3 mt-2">
              <ProgressRing
                value={compliance.overall_score}
                size={48}
                strokeWidth={4}
                color={compliance.compliant ? "text-teal" : "text-rose-ember"}
                label={`${(compliance.overall_score * 100).toFixed(0)}%`}
              />
              <div>
                <span className={`text-xs font-medium px-2 py-0.5 rounded ${compliance.compliant ? "bg-teal/10 text-teal" : "bg-rose-ember/10 text-rose-ember"}`}>
                  {compliance.compliant ? "Compliant" : "Non-Compliant"}
                </span>
                {!compliance.compliant && compliance.requirements && (
                  <p className="text-xs text-rose-ember mt-1">
                    {compliance.requirements.filter((r) => r.status !== "pass").length} requirements failing
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
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
          {lastUpdated && (
            <p className="text-xs text-fg-faint mt-2">Last verified: {lastUpdated.toLocaleString()}</p>
          )}
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
          <EmptyState title="No events recorded yet" subtitle="Record forensic events via the API to see type distribution here" />
        ) : (
          <div className="space-y-2">
            {Object.entries(data.events_by_type)
              .sort(([, a], [, b]) => b - a)
              .map(([type, count]) => {
                const maxCount = Math.max(...Object.values(data.events_by_type));
                const pct = maxCount > 0 ? (count / maxCount) * 100 : 0;
                const color = eventBarColor(type);
                return (
                  <div key={type} className="flex items-center gap-3">
                    <span className="text-xs text-fg-muted w-40 truncate" title={type.replace(/_/g, " ")}>{type.replace(/_/g, " ")}</span>
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
            <EmptyState title="No FL rounds yet" subtitle="Run a federated learning simulation to see results" />
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
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-surface-light rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${r.global_accuracy >= 0.8 ? "bg-teal" : r.global_accuracy >= 0.5 ? "bg-gold" : "bg-rose-ember"}`}
                        style={{ width: `${r.global_accuracy * 100}%` }}
                      />
                    </div>
                    <span className={`text-sm font-semibold tabular-nums ${r.global_accuracy >= 0.8 ? "text-teal" : r.global_accuracy >= 0.5 ? "text-gold" : "text-rose-ember"}`}>
                      {(r.global_accuracy * 100).toFixed(1)}%
                    </span>
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
            <EmptyState title="No models trained yet" subtitle="Train a PCA+SVM model to see it listed here" />
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
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-surface-light rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${m.accuracy >= 0.9 ? "bg-teal" : m.accuracy >= 0.7 ? "bg-gold" : "bg-rose-ember"}`}
                        style={{ width: `${m.accuracy * 100}%` }}
                      />
                    </div>
                    <span className={`text-sm font-semibold tabular-nums ${m.accuracy >= 0.9 ? "text-teal" : m.accuracy >= 0.7 ? "text-gold" : "text-rose-ember"}`}>
                      {(m.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Recent events */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-fg-secondary">Recent Events</h3>
          <Link to="/events" className="text-xs text-cyan hover:text-cyan/80 transition-colors">
            View all &rarr;
          </Link>
        </div>
        {data.recent_events.length === 0 ? (
          <EmptyState title="No events yet" subtitle="Events will appear here as they are recorded" />
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
  "rose-ember": "border-rose-ember/20 text-rose-ember",
};

function StatCard({ label, value, color = "cyan", badge }: { label: string; value: string | number; color?: string; badge?: string }) {
  return (
    <div className={`glass-card p-4 border-l-2 transition-transform duration-200 hover:scale-[1.02] ${STAT_COLORS[color] || STAT_COLORS.cyan}`}>
      <p className="text-sm text-fg-muted">{label}</p>
      <div className="flex items-center gap-2 mt-1">
        <p className={`text-2xl font-bold ${STAT_COLORS[color]?.split(" ")[1] || "text-cyan"}`}>{value}</p>
        {badge && (
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${STAT_COLORS[color]?.split(" ")[1] || "text-cyan"} bg-surface`}>
            {badge}
          </span>
        )}
      </div>
    </div>
  );
}

function EventBadge({ type }: { type: string }) {
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${eventBadgeColor(type)}`}>{type}</span>;
}
