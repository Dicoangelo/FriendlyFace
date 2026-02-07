import { useEffect, useState } from "react";

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

  // Config
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    fetch("/api/v1/fairness/status").then((r) => r.json()).then(setFairness);
    fetch("/api/v1/fairness/audits").then((r) => r.json()).then((d) => setAudits(d.audits || []));
    fetch("/api/v1/fairness/config").then((r) => r.json()).then(setConfig);
  }, []);

  const runAudit = () => {
    setError("");
    setAuditResult(null);
    let parsedGroups;
    try { parsedGroups = JSON.parse(groups); } catch { setError("Invalid JSON groups"); return; }
    fetch("/api/v1/fairness/audit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ groups: parsedGroups, demographic_parity_threshold: dpThreshold, equalized_odds_threshold: eoThreshold }),
    })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setAuditResult)
      .catch((e) => setError(e.message));
  };

  const viewDetail = (auditId: string) => {
    fetch(`/api/v1/fairness/audits/${auditId}`).then((r) => r.json()).then(setAuditDetail);
  };

  const statusColors: Record<string, string> = {
    pass: "bg-green-100 text-green-800 border-green-300",
    warning: "bg-yellow-100 text-yellow-800 border-yellow-300",
    fail: "bg-red-100 text-red-800 border-red-300",
    unknown: "bg-gray-100 text-gray-600 border-gray-300",
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Bias Audits</h2>
      {error && <div className="text-red-600 text-sm">{error}</div>}

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

      {/* Audit list */}
      {audits.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold mb-2">Audit History</h3>
          <table className="w-full text-sm">
            <thead className="text-left text-gray-500 border-b">
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
                <tr key={a.audit_id} onClick={() => viewDetail(a.audit_id)} className="border-b hover:bg-gray-50 cursor-pointer">
                  <td className="py-2 font-mono text-xs">{a.audit_id.slice(0, 8)}...</td>
                  <td className="py-2 text-gray-500">{new Date(a.timestamp).toLocaleString()}</td>
                  <td className="py-2">{a.compliant ? <span className="text-green-600">Yes</span> : <span className="text-red-600">No</span>}</td>
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
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold mb-2">Audit Detail</h3>
          <pre className="text-xs bg-gray-50 rounded p-2 overflow-x-auto">{JSON.stringify(auditDetail, null, 2)}</pre>
        </div>
      )}

      {/* Run audit form */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Run Audit</h3>
        <textarea value={groups} onChange={(e) => setGroups(e.target.value)} className="w-full border rounded px-3 py-1 text-sm font-mono h-24" placeholder="Groups JSON array" />
        <div className="flex gap-4">
          <label className="text-sm">DP Threshold <input type="number" value={dpThreshold} onChange={(e) => setDpThreshold(+e.target.value)} className="border rounded px-2 py-1 w-20 ml-1" step={0.01} /></label>
          <label className="text-sm">EO Threshold <input type="number" value={eoThreshold} onChange={(e) => setEoThreshold(+e.target.value)} className="border rounded px-2 py-1 w-20 ml-1" step={0.01} /></label>
        </div>
        <button onClick={runAudit} className="px-4 py-1 bg-blue-600 text-white rounded text-sm">Run Audit</button>
        {auditResult && <pre className="text-xs bg-gray-50 rounded p-2 overflow-x-auto">{JSON.stringify(auditResult, null, 2)}</pre>}
      </div>

      {/* Config */}
      {config && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold mb-2">Auto-Audit Config</h3>
          <pre className="text-xs">{JSON.stringify(config, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
