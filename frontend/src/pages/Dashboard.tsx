import { useEffect, useState } from "react";
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

const NAV_ICONS: Record<string, React.ReactNode> = {
  zap: <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
  eye: <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>,
  scale: <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" /></svg>,
  key: <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" /></svg>,
};

const QUICK_ACTIONS = [
  { label: "Record Event", href: "/events/live", icon: "zap", color: "text-cyan" },
  { label: "Train Model", href: "/recognition", icon: "eye", color: "text-amethyst" },
  { label: "Run Audit", href: "/bias", icon: "scale", color: "text-gold" },
  { label: "Issue VC", href: "/did", icon: "key", color: "text-teal" },
];

const STAT_ICONS: Record<string, React.ReactNode> = {
  events: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
  bundles: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" /></svg>,
  nodes: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 3v12m0 0a3 3 0 103 3 3 3 0 00-3-3zm12-6a3 3 0 10-3-3 3 3 0 003 3zm0 0v3a3 3 0 01-3 3H9" /></svg>,
  uptime: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  gallery: <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
};

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [flRounds, setFlRounds] = useState<FLRound[]>([]);
  const [models, setModels] = useState<RecognitionModel[]>([]);
  const [galleryCount, setGalleryCount] = useState<number | null>(null);
  const [compliance, setCompliance] = useState<{ compliant: boolean; overall_score: number } | null>(null);
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

    fetch("/api/v1/gallery/count")
      .then((r) => r.json())
      .then((data) => setGalleryCount(data.total ?? data.count ?? 0))
      .catch(() => {});

    fetch("/api/v1/governance/compliance")
      .then((r) => r.json())
      .then(setCompliance)
      .catch(() => {});

    return () => clearInterval(id);
  }, []);

  if (error) return (
    <div className="glass-card p-6 border-l-2 border-rose-ember/40">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-rose-ember/10 flex items-center justify-center text-rose-ember">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
        </div>
        <div>
          <p className="font-medium text-fg">Dashboard Error</p>
          <p className="text-sm text-fg-muted">Failed to load: {error}</p>
        </div>
      </div>
    </div>
  );

  if (!data) return <SkeletonDashboard />;

  return (
    <div className="space-y-6">
      {/* Quick actions bar */}
      <div className="glass-card p-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-fg">Welcome back</h2>
            <p className="text-sm text-fg-muted">
              {data.storage_backend === "sqlite" ? "SQLite" : "Supabase"} backend
              {" "}&middot;{" "}
              {data.total_events} events tracked
            </p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {QUICK_ACTIONS.map((action) => (
              <a
                key={action.label}
                href={action.href}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-border-theme text-fg-secondary hover:bg-fg/5 hover:text-fg transition-all"
              >
                <span className={action.color}>{NAV_ICONS[action.icon]}</span>
                {action.label}
              </a>
            ))}
          </div>
        </div>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard label="Total Events" value={data.total_events} color="cyan" icon={STAT_ICONS.events} />
        <StatCard label="Total Bundles" value={data.total_bundles} color="amethyst" icon={STAT_ICONS.bundles} />
        <StatCard label="Provenance Nodes" value={data.total_provenance_nodes} color="teal" icon={STAT_ICONS.nodes} />
        <StatCard
          label="Uptime"
          value={`${Math.floor(data.uptime_seconds / 60)}m ${Math.floor(data.uptime_seconds % 60)}s`}
          color="gold"
          icon={STAT_ICONS.uptime}
        />
        {galleryCount !== null && (
          <StatCard label="Gallery Subjects" value={galleryCount} color="cyan" icon={STAT_ICONS.gallery} />
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
              <span className={`text-xs font-medium px-2 py-0.5 rounded ${compliance.compliant ? "bg-teal/10 text-teal" : "bg-rose-ember/10 text-rose-ember"}`}>
                {compliance.compliant ? "Compliant" : "Non-Compliant"}
              </span>
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
        <h3 className="font-semibold text-fg-secondary mb-3">Recent Events</h3>
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

const STAT_ICON_BG: Record<string, string> = {
  cyan: "bg-cyan/10",
  amethyst: "bg-amethyst/10",
  teal: "bg-teal/10",
  gold: "bg-gold/10",
  "rose-ember": "bg-rose-ember/10",
};

function StatCard({ label, value, color = "cyan", icon }: { label: string; value: string | number; color?: string; icon?: React.ReactNode }) {
  const textColor = STAT_COLORS[color]?.split(" ")[1] || "text-cyan";
  return (
    <div className={`glass-card p-4 border-l-2 transition-transform duration-200 hover:scale-[1.02] ${STAT_COLORS[color] || STAT_COLORS.cyan}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-fg-muted">{label}</p>
          <p className={`text-2xl font-bold mt-1 tabular-nums ${textColor}`}>{value}</p>
        </div>
        {icon && (
          <div className={`w-9 h-9 rounded-lg ${STAT_ICON_BG[color] || "bg-cyan/10"} flex items-center justify-center ${textColor} flex-shrink-0`}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}

function EventBadge({ type }: { type: string }) {
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${eventBadgeColor(type)}`}>{type}</span>;
}
