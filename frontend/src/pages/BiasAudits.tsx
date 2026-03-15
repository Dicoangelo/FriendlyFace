import { useEffect, useState } from "react";
import LoadingButton from "../components/LoadingButton";
import ProgressRing from "../components/ProgressRing";
import { SkeletonCard } from "../components/Skeleton";
import { statusColor } from "../constants/eventColors";

interface AuditSummary {
  audit_id: string;
  timestamp: string;
  compliant: boolean;
  fairness_score: number;
  demographic_parity_gap: number;
  equalized_odds_gap: number;
}

interface FairnessStatus {
  status: string;
  fairness_score?: number;
  total_audits: number;
  compliant_audits?: number;
}

interface DemographicGroup {
  group_name: string;
  total_results: number;
  accuracy?: number;
}

interface GroupRow {
  group_name: string;
  true_positives: number;
  false_positives: number;
  true_negatives: number;
  false_negatives: number;
}

interface AuditResult {
  audit_id: string;
  compliant: boolean;
  fairness_score: number | null;
  demographic_parity_gap: number;
  equalized_odds_gap: number;
  status?: string;
}

const DEFAULT_GROUP: GroupRow = {
  group_name: "",
  true_positives: 0,
  false_positives: 0,
  true_negatives: 0,
  false_negatives: 0,
};

function scoreColor(score: number): string {
  if (score >= 0.8) return "text-teal";
  if (score >= 0.5) return "text-gold";
  return "text-rose-ember";
}

export default function BiasAudits() {
  const [fairness, setFairness] = useState<FairnessStatus | null>(null);
  const [fairnessLoading, setFairnessLoading] = useState(true);
  const [audits, setAudits] = useState<AuditSummary[]>([]);
  const [expandedAudit, setExpandedAudit] = useState<string | null>(null);
  const [auditDetail, setAuditDetail] = useState<Record<string, unknown> | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Dynamic group builder
  const [groups, setGroups] = useState<GroupRow[]>([
    { group_name: "A", true_positives: 80, false_positives: 10, true_negatives: 90, false_negatives: 20 },
    { group_name: "B", true_positives: 70, false_positives: 15, true_negatives: 85, false_negatives: 30 },
  ]);
  const [dpThreshold, setDpThreshold] = useState(0.1);
  const [eoThreshold, setEoThreshold] = useState(0.1);
  const [auditResult, setAuditResult] = useState<AuditResult | null>(null);
  const [error, setError] = useState("");
  const [auditLoading, setAuditLoading] = useState(false);

  // Demographics
  const [demographics, setDemographics] = useState<DemographicGroup[]>([]);

  useEffect(() => {
    setFairnessLoading(true);
    fetch("/api/v1/fairness/status")
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setFairness)
      .catch(() => {})
      .finally(() => setFairnessLoading(false));

    fetch("/api/v1/fairness/audits")
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d) => setAudits(d.items || []))
      .catch(() => {});

    fetch("/api/v1/fairness/demographics")
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d) => setDemographics(d.groups || []))
      .catch(() => {});
  }, []);

  const refreshStatus = () => {
    fetch("/api/v1/fairness/status")
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setFairness)
      .catch(() => {});
    fetch("/api/v1/fairness/audits")
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d) => setAudits(d.items || []))
      .catch(() => {});
  };

  const updateGroup = (idx: number, field: keyof GroupRow, value: string | number) => {
    setGroups((prev) => prev.map((g, i) => (i === idx ? { ...g, [field]: value } : g)));
  };

  const addGroup = () => setGroups((prev) => [...prev, { ...DEFAULT_GROUP, group_name: `Group ${prev.length + 1}` }]);

  const removeGroup = (idx: number) => {
    if (groups.length <= 2) return;
    setGroups((prev) => prev.filter((_, i) => i !== idx));
  };

  const runAudit = () => {
    setError("");
    setAuditResult(null);

    const invalid = groups.find((g) => !g.group_name.trim());
    if (invalid) { setError("All groups must have a name"); return; }

    setAuditLoading(true);
    fetch("/api/v1/fairness/audit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ groups, demographic_parity_threshold: dpThreshold, equalized_odds_threshold: eoThreshold }),
    })
      .then((r) => { if (!r.ok) throw new Error(`Audit failed (${r.status})`); return r.json(); })
      .then((data) => {
        setAuditResult(data as AuditResult);
        refreshStatus();
      })
      .catch((e) => setError(e.message))
      .finally(() => setAuditLoading(false));
  };

  const toggleDetail = (auditId: string) => {
    if (expandedAudit === auditId) {
      setExpandedAudit(null);
      setAuditDetail(null);
      return;
    }
    setExpandedAudit(auditId);
    setAuditDetail(null);
    setDetailLoading(true);
    fetch(`/api/v1/fairness/audits/${auditId}`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setAuditDetail)
      .catch((e) => setError(e.message))
      .finally(() => setDetailLoading(false));
  };

  const numFields: (keyof GroupRow)[] = ["true_positives", "false_positives", "true_negatives", "false_negatives"];

  return (
    <div className="space-y-6">
      {/* Error banner */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError("")} className="ml-2 hover:text-rose-ember/70">&times;</button>
        </div>
      )}

      {/* Fairness status card */}
      {fairnessLoading ? (
        <SkeletonCard className="h-28" />
      ) : fairness ? (
        <div className={`rounded-lg border-2 p-5 ${statusColor(fairness.status)}`}>
          <div className="flex items-center gap-6">
            {fairness.fairness_score !== undefined && (
              <ProgressRing
                value={fairness.fairness_score}
                size={72}
                strokeWidth={6}
                color={scoreColor(fairness.fairness_score)}
                label={`${(fairness.fairness_score * 100).toFixed(0)}%`}
              />
            )}
            <div className="flex-1">
              <p className="text-lg font-bold uppercase">{fairness.status}</p>
              {fairness.fairness_score !== undefined && (
                <p className="text-sm text-fg-muted">Fairness Score: {fairness.fairness_score.toFixed(3)}</p>
              )}
            </div>
            <div className="text-sm text-right space-y-1">
              <p>Total audits: <span className="font-semibold">{fairness.total_audits}</span></p>
              {fairness.compliant_audits !== undefined && (
                <p>Compliant: <span className="font-semibold">{fairness.compliant_audits}</span></p>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {/* Run audit form */}
      <div className="glass-card p-4 space-y-4">
        <h3 className="font-semibold text-fg-secondary">Run Audit</h3>

        {/* Dynamic group builder */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-fg-muted">Demographic Groups ({groups.length})</span>
            <button onClick={addGroup} className="btn-secondary text-xs px-2 py-1">+ Add Group</button>
          </div>
          <div className="space-y-2">
            {groups.map((g, idx) => (
              <div key={idx} className="flex items-center gap-2 bg-surface/50 rounded-lg p-2">
                <input
                  type="text"
                  value={g.group_name}
                  onChange={(e) => updateGroup(idx, "group_name", e.target.value)}
                  placeholder="Group name"
                  className="ff-input w-28 text-sm"
                />
                {numFields.map((field) => (
                  <label key={field} className="flex flex-col items-center gap-0.5">
                    <span className="text-[10px] text-fg-faint uppercase">{field.replace(/_/g, " ").replace("true ", "T").replace("false ", "F")}</span>
                    <input
                      type="number"
                      value={g[field]}
                      onChange={(e) => updateGroup(idx, field, parseInt(e.target.value) || 0)}
                      className="ff-input w-16 text-sm text-center"
                      min={0}
                    />
                  </label>
                ))}
                <button
                  onClick={() => removeGroup(idx)}
                  disabled={groups.length <= 2}
                  className="text-fg-faint hover:text-rose-ember disabled:opacity-30 ml-auto text-lg"
                  title="Remove group"
                >
                  &times;
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Threshold sliders */}
        <div className="flex flex-wrap gap-6">
          <label className="flex flex-col gap-1">
            <span className="text-sm text-fg-secondary">DP Threshold: {dpThreshold.toFixed(2)}</span>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={dpThreshold}
              onChange={(e) => setDpThreshold(parseFloat(e.target.value))}
              className="w-48 accent-amethyst"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-sm text-fg-secondary">EO Threshold: {eoThreshold.toFixed(2)}</span>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={eoThreshold}
              onChange={(e) => setEoThreshold(parseFloat(e.target.value))}
              className="w-48 accent-amethyst"
            />
          </label>
        </div>

        <LoadingButton onClick={runAudit} loading={auditLoading} loadingText="Auditing...">Run Audit</LoadingButton>

        {/* Audit result card */}
        {auditResult && (
          <div className={`rounded-lg border p-4 ${auditResult.compliant ? "bg-teal/5 border-teal/20" : "bg-rose-ember/5 border-rose-ember/20"}`}>
            <div className="flex items-center gap-4">
              <span className={`text-2xl font-bold ${auditResult.compliant ? "text-teal" : "text-rose-ember"}`}>
                {auditResult.compliant ? "Compliant" : "Non-Compliant"}
              </span>
              {auditResult.fairness_score != null && (
                <ProgressRing
                  value={auditResult.fairness_score}
                  size={56}
                  strokeWidth={5}
                  color={scoreColor(auditResult.fairness_score)}
                  label={`${(auditResult.fairness_score * 100).toFixed(0)}%`}
                />
              )}
            </div>
            <div className="mt-2 flex gap-6 text-sm text-fg-muted">
              <span>DP Gap: <span className="font-mono font-semibold">{auditResult.demographic_parity_gap.toFixed(4)}</span></span>
              <span>EO Gap: <span className="font-mono font-semibold">{auditResult.equalized_odds_gap.toFixed(4)}</span></span>
            </div>
          </div>
        )}
      </div>

      {/* Audit history table */}
      {audits.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-2">Audit History</h3>
          <table className="w-full text-sm">
            <thead className="text-left text-fg-muted border-b border-border-theme">
              <tr>
                <th className="pb-2">ID</th>
                <th className="pb-2">Timestamp</th>
                <th className="pb-2">Compliant</th>
                <th className="pb-2">Fairness Score</th>
                <th className="pb-2">DP Gap</th>
                <th className="pb-2">EO Gap</th>
              </tr>
            </thead>
            <tbody>
              {audits.map((a) => (
                <>
                  <tr
                    key={a.audit_id}
                    onClick={() => toggleDetail(a.audit_id)}
                    className={`border-b border-border-theme hover:bg-fg/[0.02] cursor-pointer ${expandedAudit === a.audit_id ? "bg-fg/[0.03]" : ""}`}
                  >
                    <td className="py-2 font-mono text-xs">{a.audit_id.slice(0, 8)}...</td>
                    <td className="py-2 text-fg-muted">{new Date(a.timestamp).toLocaleString()}</td>
                    <td className="py-2">
                      {a.compliant ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-teal/10 text-teal">Yes</span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-rose-ember/10 text-rose-ember">No</span>
                      )}
                    </td>
                    <td className="py-2">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 max-w-[100px] h-2 bg-surface rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${a.fairness_score >= 0.8 ? "bg-teal" : a.fairness_score >= 0.5 ? "bg-gold" : "bg-rose-ember"}`}
                            style={{ width: `${(a.fairness_score ?? 0) * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs">{a.fairness_score?.toFixed(3)}</span>
                      </div>
                    </td>
                    <td className="py-2 font-mono text-xs">{a.demographic_parity_gap.toFixed(4)}</td>
                    <td className="py-2 font-mono text-xs">{a.equalized_odds_gap.toFixed(4)}</td>
                  </tr>
                  {expandedAudit === a.audit_id && (
                    <tr key={`${a.audit_id}-detail`}>
                      <td colSpan={6} className="p-3 bg-surface/30">
                        {detailLoading ? (
                          <div className="animate-pulse h-20 bg-surface rounded-lg" />
                        ) : auditDetail ? (
                          <pre className="text-xs bg-surface rounded-lg p-3 overflow-x-auto max-h-64 overflow-y-auto">
                            {JSON.stringify(auditDetail, null, 2)}
                          </pre>
                        ) : null}
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Demographics breakdown table */}
      {demographics.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-3">Demographics Breakdown</h3>
          <table className="w-full text-sm">
            <thead className="text-left text-fg-muted border-b border-border-theme">
              <tr>
                <th className="pb-2">Group</th>
                <th className="pb-2">Total Results</th>
                <th className="pb-2">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {demographics.map((g) => {
                const acc = g.accuracy ?? 0;
                const color = acc >= 0.8 ? "teal" : acc >= 0.5 ? "gold" : "rose-ember";
                return (
                  <tr key={g.group_name} className="border-b border-border-theme">
                    <td className="py-2">{g.group_name}</td>
                    <td className="py-2 text-fg-muted">{g.total_results}</td>
                    <td className="py-2">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 max-w-[120px] h-2 bg-surface rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full bg-${color}`}
                            style={{ width: `${Math.min(acc * 100, 100)}%` }}
                          />
                        </div>
                        <span className={`font-mono text-xs text-${color}`}>{(acc * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
