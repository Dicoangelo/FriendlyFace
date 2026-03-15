import { useEffect, useState } from "react";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import { SkeletonCard, SkeletonRow } from "../components/Skeleton";

interface PolicyConfig {
  max_age_days: number;
  max_events: number;
  action: string;
}

interface PolicyRun {
  timestamp: string;
  events_processed: number;
  events_retained: number;
  events_expired: number;
  dry_run?: boolean;
}

export default function RetentionPolicies() {
  const [policy, setPolicy] = useState<PolicyConfig | null>(null);
  const [loadingPolicy, setLoadingPolicy] = useState(true);
  const [error, setError] = useState("");

  // Edit form
  const [editing, setEditing] = useState(false);
  const [maxAgeDays, setMaxAgeDays] = useState(365);
  const [maxEvents, setMaxEvents] = useState(10000);
  const [action, setAction] = useState("archive");
  const [saving, setSaving] = useState(false);

  // Apply
  const [applying, setApplying] = useState(false);
  const [applyResult, setApplyResult] = useState<PolicyRun | null>(null);
  const [dryRun, setDryRun] = useState(false);

  // History
  const [history, setHistory] = useState<PolicyRun[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);

  const fetchPolicy = () => {
    setLoadingPolicy(true);
    fetch("/api/v1/retention/policies")
      .then((r) => r.json())
      .then((data) => {
        const policies = data.policies || [];
        const active = policies.find((p: Record<string, unknown>) => p.enabled) || policies[0];
        if (active) {
          const p = {
            max_age_days: active.retention_days ?? active.max_age_days ?? 365,
            max_events: active.max_events ?? 10000,
            action: active.action ?? "archive",
          };
          setPolicy(p);
          setMaxAgeDays(p.max_age_days);
          setMaxEvents(p.max_events);
          setAction(p.action);
        } else {
          setPolicy({ max_age_days: 365, max_events: 10000, action: "archive" });
        }
        setLoadingPolicy(false);
      })
      .catch(() => {
        setPolicy({ max_age_days: 365, max_events: 10000, action: "archive" });
        setLoadingPolicy(false);
      });
  };

  const fetchHistory = () => {
    setLoadingHistory(true);
    fetch("/api/v1/retention/evaluate/history")
      .then((r) => r.json())
      .then((data) => {
        setHistory(data.runs || data.history || []);
        setLoadingHistory(false);
      })
      .catch(() => setLoadingHistory(false));
  };

  useEffect(() => {
    fetchPolicy();
    fetchHistory();
  }, []);

  const savePolicy = () => {
    setError("");
    setSaving(true);
    fetch("/api/v1/retention/policies", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: `Policy ${new Date().toISOString().slice(0, 10)}`,
        entity_type: "event",
        retention_days: maxAgeDays,
        action,
        enabled: true,
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(() => {
        setPolicy({ max_age_days: maxAgeDays, max_events: maxEvents, action });
        setEditing(false);
        setSaving(false);
      })
      .catch((e) => {
        setError(`Save error: ${e.message}`);
        setSaving(false);
      });
  };

  const applyPolicy = () => {
    setError("");
    setApplyResult(null);
    setApplying(true);

    const params = dryRun ? "?dry_run=true" : "";
    fetch(`/api/v1/retention/evaluate${params}`, { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setApplyResult(data);
        setApplying(false);
        fetchHistory();
      })
      .catch((e) => {
        setError(`Apply error: ${e.message}`);
        setApplying(false);
      });
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Active policy card */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-fg-secondary">Active Policy</h3>
          {!editing && (
            <button onClick={() => setEditing(true)} className="text-xs text-cyan hover:text-cyan/80 px-2 py-1 rounded hover:bg-cyan/10 transition-colors">
              Edit
            </button>
          )}
        </div>

        {loadingPolicy ? (
          <SkeletonCard className="h-20" />
        ) : editing ? (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-4 items-end">
              <label className="text-sm text-fg-secondary">
                Max Age (days)
                <input
                  type="number"
                  value={maxAgeDays}
                  onChange={(e) => setMaxAgeDays(+e.target.value)}
                  className="ff-input w-28 block mt-1"
                  min={1}
                />
              </label>
              <label className="text-sm text-fg-secondary">
                Max Events
                <input
                  type="number"
                  value={maxEvents}
                  onChange={(e) => setMaxEvents(+e.target.value)}
                  className="ff-input w-28 block mt-1"
                  min={1}
                />
              </label>
              <label className="text-sm text-fg-secondary">
                Action
                <select
                  value={action}
                  onChange={(e) => setAction(e.target.value)}
                  className="ff-select block mt-1"
                >
                  <option value="archive">archive</option>
                  <option value="delete">delete</option>
                  <option value="flag">flag</option>
                </select>
              </label>
            </div>
            <div className="flex gap-2">
              <LoadingButton onClick={savePolicy} loading={saving} loadingText="Saving...">
                Save Policy
              </LoadingButton>
              <button onClick={() => setEditing(false)} className="btn-ghost">Cancel</button>
            </div>
          </div>
        ) : policy ? (
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-fg-muted">Max Age</p>
              <p className="text-lg font-bold text-cyan">{policy.max_age_days} days</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Max Events</p>
              <p className="text-lg font-bold text-amethyst">{policy.max_events}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Action</p>
              <p className="text-lg font-bold text-gold">{policy.action}</p>
            </div>
          </div>
        ) : null}
      </div>

      {/* Apply policy */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Apply Policy</h3>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-fg-secondary cursor-pointer">
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(e) => setDryRun(e.target.checked)}
              className="rounded border-border-theme"
            />
            Dry run (preview only)
          </label>
          <LoadingButton
            onClick={applyPolicy}
            loading={applying}
            className={dryRun ? "btn-primary" : "btn-accent"}
            loadingText="Applying..."
          >
            {dryRun ? "Preview Policy" : "Apply Policy Now"}
          </LoadingButton>
        </div>

        {applyResult && (
          <div className="bg-teal/10 border border-teal/20 rounded-lg p-3 animate-fade-in">
            {applyResult.dry_run && (
              <p className="text-xs text-gold font-medium mb-2">Dry Run — no changes applied</p>
            )}
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-xs text-fg-muted">Processed</p>
                <p className="text-lg font-bold text-cyan">{applyResult.events_processed}</p>
              </div>
              <div>
                <p className="text-xs text-fg-muted">Retained</p>
                <p className="text-lg font-bold text-teal">{applyResult.events_retained}</p>
              </div>
              <div>
                <p className="text-xs text-fg-muted">Expired</p>
                <p className="text-lg font-bold text-rose-ember">{applyResult.events_expired}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Policy history */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Policy Run History</h3>
        {loadingHistory ? (
          <div className="space-y-1">
            {[...Array(3)].map((_, i) => <SkeletonRow key={i} />)}
          </div>
        ) : history.length === 0 ? (
          <EmptyState title="No policy runs" subtitle="Apply a policy to see run history here" />
        ) : (
          <div className="space-y-2">
            {history.map((run, i) => (
              <div key={i} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                <div className="flex items-center gap-3">
                  <p className="text-xs text-fg-faint">{run.timestamp ? new Date(run.timestamp).toLocaleString() : "—"}</p>
                  {run.dry_run && (
                    <span className="px-1.5 py-0.5 rounded text-xs bg-gold/10 text-gold">dry-run</span>
                  )}
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <span className="text-fg-muted">{run.events_processed} processed</span>
                  <span className="text-teal">{run.events_retained} retained</span>
                  <span className="text-rose-ember">{run.events_expired} expired</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
