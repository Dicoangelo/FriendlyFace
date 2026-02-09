import { useEffect, useState } from "react";
import EmptyState from "../components/EmptyState";

interface RetentionPolicy {
  policy_id: string;
  name: string;
  entity_type: string;
  retention_days: number;
  action: string;
  enabled: boolean;
  created_at: string;
}

export default function RetentionPolicies() {
  const [policies, setPolicies] = useState<RetentionPolicy[]>([]);
  const [error, setError] = useState("");

  // Create form
  const [name, setName] = useState("");
  const [entityType, setEntityType] = useState("consent");
  const [retentionDays, setRetentionDays] = useState(365);
  const [action, setAction] = useState("erase");
  const [creating, setCreating] = useState(false);
  const [createResult, setCreateResult] = useState<Record<string, unknown> | null>(null);

  // Evaluate
  const [evaluating, setEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState<Record<string, unknown> | null>(null);

  const fetchPolicies = () => {
    fetch("/api/v1/retention/policies")
      .then((r) => r.json())
      .then((data) => setPolicies(data.policies || []))
      .catch(() => {});
  };

  useEffect(() => {
    fetchPolicies();
  }, []);

  const handleCreate = () => {
    if (!name.trim()) {
      setError("Policy name is required");
      return;
    }
    setError("");
    setCreateResult(null);
    setCreating(true);

    fetch("/api/v1/retention/policies", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: name.trim(),
        entity_type: entityType,
        retention_days: retentionDays,
        action,
        enabled: true,
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setCreateResult(data);
        setCreating(false);
        setName("");
        fetchPolicies();
      })
      .catch((e) => {
        setError(`Create error: ${e.message}`);
        setCreating(false);
      });
  };

  const deletePolicy = (policyId: string) => {
    fetch(`/api/v1/retention/policies/${policyId}`, { method: "DELETE" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        fetchPolicies();
      })
      .catch((e) => setError(`Delete error: ${e.message}`));
  };

  const handleEvaluate = () => {
    setEvaluating(true);
    setEvalResult(null);
    setError("");

    fetch("/api/v1/retention/evaluate", { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setEvalResult(data);
        setEvaluating(false);
      })
      .catch((e) => {
        setError(`Evaluate error: ${e.message}`);
        setEvaluating(false);
      });
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Create policy */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Create Retention Policy</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Name
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="ff-input w-48 block mt-1"
              placeholder="e.g. Consent 1yr"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Entity Type
            <select
              value={entityType}
              onChange={(e) => setEntityType(e.target.value)}
              className="ff-select block mt-1"
            >
              <option value="consent">consent</option>
              <option value="subject">subject</option>
              <option value="event">event</option>
              <option value="model">model</option>
            </select>
          </label>
          <label className="text-sm text-fg-secondary">
            Retention Days
            <input
              type="number"
              value={retentionDays}
              onChange={(e) => setRetentionDays(+e.target.value)}
              className="ff-input w-24 block mt-1"
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
              <option value="erase">erase</option>
            </select>
          </label>
        </div>
        <button onClick={handleCreate} disabled={creating} className="btn-primary">
          {creating ? "Creating..." : "Create Policy"}
        </button>
        {createResult && (
          <div className="bg-teal/10 border border-teal/20 rounded-lg px-3 py-2 text-teal text-sm">
            Policy created: {String(createResult.name || createResult.policy_id)}
          </div>
        )}
      </div>

      {/* Evaluate */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Evaluate Policies</h3>
        <p className="text-xs text-fg-muted">Run all enabled retention policies and erase expired data.</p>
        <button onClick={handleEvaluate} disabled={evaluating} className="btn-accent">
          {evaluating ? "Evaluating..." : "Run Evaluation"}
        </button>
        {evalResult && (
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
            {JSON.stringify(evalResult, null, 2)}
          </pre>
        )}
      </div>

      {/* Policies list */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">
          Active Policies <span className="text-fg-faint font-normal">({policies.length})</span>
        </h3>
        {policies.length === 0 ? (
          <EmptyState title="No retention policies" subtitle="Create a policy to automatically manage data lifecycle" />
        ) : (
          <div className="space-y-2">
            {policies.map((p) => (
              <div key={p.policy_id} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                <div>
                  <div className="flex items-center gap-2">
                    <p className="text-sm text-fg-secondary font-medium">{p.name}</p>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${p.enabled ? "bg-teal/10 text-teal" : "bg-fg/5 text-fg-faint"}`}>
                      {p.enabled ? "Enabled" : "Disabled"}
                    </span>
                  </div>
                  <p className="text-xs text-fg-faint">
                    {p.entity_type} — {p.retention_days} days — action: {p.action}
                  </p>
                </div>
                <button
                  onClick={() => deletePolicy(p.policy_id)}
                  className="text-xs text-rose-ember hover:text-rose-ember/80 px-2 py-1 rounded hover:bg-rose-ember/10 transition-colors"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
