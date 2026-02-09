import { useEffect, useState } from "react";
import LoadingButton from "../components/LoadingButton";

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

export default function BiasAudits() {
  const [fairness, setFairness] = useState<FairnessStatus | null>(null);
  const [audits, setAudits] = useState<AuditSummary[]>([]);
  const [auditDetail, setAuditDetail] = useState<Record<string, unknown> | null>(null);

  // Run audit form
  const [groups, setGroups] = useState('[{"group_name":"A","true_positives":80,"false_positives":10,"true_negatives":90,"false_negatives":20},{"group_name":"B","true_positives":70,"false_positives":15,"true_negatives":85,"false_negatives":30}]');
  const [dpThreshold, setDpThreshold] = useState(0.1);
  const [eoThreshold, setEoThreshold] = useState(0.1);
  const [auditResult, setAuditResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [auditLoading, setAuditLoading] = useState(false);

  // Config
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);

  // Demographics
  const [demographics, setDemographics] = useState<DemographicGroup[]>([]);

  useEffect(() => {
    fetch("/api/v1/fairness/status").then((r) => r.json()).then(setFairness);
    fetch("/api/v1/fairness/audits").then((r) => r.json()).then((d) => setAudits(d.items || []));
    fetch("/api/v1/fairness/config").then((r) => r.json()).then(setConfig);
    fetch("/api/v1/fairness/demographics").then((r) => r.json()).then((d) => setDemographics(d.groups || [])).catch(() => {});
  }, []);

  const runAudit = () => {
    setError("");
    setAuditResult(null);
    let parsedGroups;
    try { parsedGroups = JSON.parse(groups); } catch { setError("Invalid JSON groups"); return; }
    setAuditLoading(true);
    fetch("/api/v1/fairness/audit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ groups: parsedGroups, demographic_parity_threshold: dpThreshold, equalized_odds_threshold: eoThreshold }),
    })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setAuditResult)
      .catch((e) => setError(e.message))
      .finally(() => setAuditLoading(false));
  };

  const viewDetail = (auditId: string) => {
    fetch(`/api/v1/fairness/audits/${auditId}`).then((r) => r.json()).then(setAuditDetail);
  };

  const statusColors: Record<string, string> = {
    pass: "bg-teal/10 text-teal border-teal/20",
    warning: "bg-gold/10 text-gold border-gold/20",
    fail: "bg-rose-ember/10 text-rose-ember border-rose-ember/20",
    unknown: "bg-fg/5 text-fg-muted border-border-theme",
  };

  return (
    <div className="space-y-6">
      {/* Page title shown in header bar */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Status banner */}
      {fairness && (
        <div className={`rounded-lg border-2 p-4 ${statusColors[fairness.status] || statusColors.unknown}`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-lg font-bold uppercase">{fairness.status}</p>
              {fairness.fairness_score !== undefined && (
                <p className="text-sm">Score: {fairness.fairness_score.toFixed(3)}</p>
              )}
            </div>
            <div className="text-sm text-right">
              <p>Total audits: {fairness.total_audits}</p>
              {fairness.compliant_audits !== undefined && <p>Compliant: {fairness.compliant_audits}</p>}
            </div>
          </div>
        </div>
      )}

      {/* Demographic stats */}
      {demographics.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-3">Demographic Accuracy</h3>
          <div className="space-y-2">
            {demographics.map((g) => {
              const acc = g.accuracy ?? 0;
              const pct = Math.min(acc * 100, 100);
              const color = acc >= 0.8 ? "teal" : acc >= 0.5 ? "gold" : "rose-ember";
              return (
                <div key={g.group_name} className="flex items-center gap-3">
                  <span className="text-xs text-fg-muted w-32 truncate">{g.group_name}</span>
                  <div className="flex-1 h-6 bg-surface rounded-md overflow-hidden relative">
                    <div
                      className={`h-full rounded-md bg-${color}/30 transition-all duration-500`}
                      style={{ width: `${pct}%` }}
                    />
                    <span className={`absolute inset-y-0 right-2 flex items-center text-xs font-semibold text-${color}`}>
                      {(acc * 100).toFixed(1)}%
                    </span>
                  </div>
                  <span className="text-xs text-fg-faint w-16 text-right">{g.total_results} results</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Audit list */}
      {audits.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-2">Audit History</h3>
          <table className="w-full text-sm">
            <thead className="text-left text-fg-muted border-b border-border-theme">
              <tr>
                <th className="pb-2">ID</th>
                <th className="pb-2">Timestamp</th>
                <th className="pb-2">Compliant</th>
                <th className="pb-2">Score</th>
                <th className="pb-2">DP Gap</th>
                <th className="pb-2">EO Gap</th>
              </tr>
            </thead>
            <tbody>
              {audits.map((a) => (
                <tr key={a.audit_id} onClick={() => viewDetail(a.audit_id)} className="border-b border-border-theme hover:bg-fg/[0.02] cursor-pointer">
                  <td className="py-2 font-mono text-xs">{a.audit_id.slice(0, 8)}...</td>
                  <td className="py-2 text-fg-muted">{new Date(a.timestamp).toLocaleString()}</td>
                  <td className="py-2">{a.compliant ? <span className="text-teal">Yes</span> : <span className="text-rose-ember">No</span>}</td>
                  <td className="py-2">{a.fairness_score?.toFixed(3)}</td>
                  <td className="py-2">{a.demographic_parity_gap.toFixed(4)}</td>
                  <td className="py-2">{a.equalized_odds_gap.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {auditDetail && (
        <div className="glass-card p-4">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="font-semibold text-fg-secondary">Audit Detail</h3>
            <AuditSourceBadge detail={auditDetail} />
          </div>
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">{JSON.stringify(auditDetail, null, 2)}</pre>
        </div>
      )}

      {/* Run audit form */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Run Audit</h3>
        <textarea value={groups} onChange={(e) => setGroups(e.target.value)} className="w-full ff-textarea font-mono h-24" placeholder="Groups JSON array" />
        <div className="flex gap-4">
          <label className="text-sm text-fg-secondary">DP Threshold <input type="number" value={dpThreshold} onChange={(e) => setDpThreshold(+e.target.value)} className="ff-input w-20 ml-1" step={0.01} /></label>
          <label className="text-sm text-fg-secondary">EO Threshold <input type="number" value={eoThreshold} onChange={(e) => setEoThreshold(+e.target.value)} className="ff-input w-20 ml-1" step={0.01} /></label>
        </div>
        <LoadingButton onClick={runAudit} loading={auditLoading} loadingText="Auditing...">Run Audit</LoadingButton>
        {auditResult && <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">{JSON.stringify(auditResult, null, 2)}</pre>}
      </div>

      {/* Config */}
      {config && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-2">Auto-Audit Config</h3>
          <pre className="text-xs">{JSON.stringify(config, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

function AuditSourceBadge({ detail }: { detail: Record<string, unknown> }) {
  const metadata = detail.metadata as Record<string, unknown> | undefined;
  const source = metadata?.source as string | undefined;
  if (source === "real") {
    return (
      <span className="px-2 py-0.5 rounded text-xs font-medium bg-teal/10 text-teal border border-teal/20">
        Real Data
      </span>
    );
  }
  return (
    <span className="px-2 py-0.5 rounded text-xs font-medium bg-gold/10 text-gold border border-gold/20">
      Synthetic Data
    </span>
  );
}
