import { useState, useCallback } from "react";
import { useToast } from "../components/Toast";
import ConfirmDialog from "../components/ConfirmDialog";

export default function ConsentManagement() {
  const toast = useToast();

  // Grant consent
  const [subjectId, setSubjectId] = useState("");
  const [purpose, setPurpose] = useState("recognition");
  const [expiry, setExpiry] = useState("");
  const [grantResult, setGrantResult] = useState<Record<string, unknown> | null>(null);

  // Check consent
  const [checkSubject, setCheckSubject] = useState("");
  const [checkPurpose, setCheckPurpose] = useState("recognition");
  const [checkResult, setCheckResult] = useState<Record<string, unknown> | null>(null);

  // History
  const [historySubject, setHistorySubject] = useState("");
  const [history, setHistory] = useState<{ records: Array<Record<string, unknown>> } | null>(null);

  // Revoke
  const [revokeSubject, setRevokeSubject] = useState("");
  const [revokePurpose, setRevokePurpose] = useState("recognition");
  const [revokeReason, setRevokeReason] = useState("");
  const [revokeResult, setRevokeResult] = useState<Record<string, unknown> | null>(null);

  // Confirm dialog
  const [confirmOpen, setConfirmOpen] = useState(false);

  const [error, setError] = useState("");

  const grantConsent = () => {
    setError("");
    setGrantResult(null);
    fetch("/api/v1/consent/grant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject_id: subjectId, purpose, expiry: expiry || null }),
    })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => {
        setGrantResult(data);
        toast.success(`Consent granted for subject "${subjectId}"`);
      })
      .catch((e) => {
        setError(e.message);
        toast.error(`Failed to grant consent: ${e.message}`);
      });
  };

  const checkConsent = () => {
    setError("");
    setCheckResult(null);
    fetch("/api/v1/consent/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject_id: checkSubject, purpose: checkPurpose }),
    })
      .then((r) => r.json())
      .then(setCheckResult)
      .catch((e) => {
        setError(e.message);
        toast.error(`Failed to check consent: ${e.message}`);
      });
  };

  const fetchHistory = () => {
    setError("");
    setHistory(null);
    fetch(`/api/v1/consent/history/${encodeURIComponent(historySubject)}`)
      .then((r) => r.json())
      .then(setHistory)
      .catch((e) => {
        setError(e.message);
        toast.error(`Failed to fetch history: ${e.message}`);
      });
  };

  const executeRevoke = useCallback(() => {
    setConfirmOpen(false);
    setError("");
    setRevokeResult(null);
    fetch("/api/v1/consent/revoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject_id: revokeSubject, purpose: revokePurpose, reason: revokeReason }),
    })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => {
        setRevokeResult(data);
        toast.warning(`Consent revoked for subject "${revokeSubject}"`);
      })
      .catch((e) => {
        setError(e.message);
        toast.error(`Failed to revoke consent: ${e.message}`);
      });
  }, [revokeSubject, revokePurpose, revokeReason, toast]);

  const handleRevokeClick = () => {
    setConfirmOpen(true);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Consent Management</h2>
      {error && <div className="text-red-600 text-sm">{error}</div>}

      {/* Grant */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Grant Consent</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <input type="text" placeholder="Subject ID" value={subjectId} onChange={(e) => setSubjectId(e.target.value)} className="border rounded px-3 py-1 text-sm" />
          <input type="text" placeholder="Purpose" value={purpose} onChange={(e) => setPurpose(e.target.value)} className="border rounded px-3 py-1 text-sm" />
          <input type="text" placeholder="Expiry (ISO-8601, optional)" value={expiry} onChange={(e) => setExpiry(e.target.value)} className="border rounded px-3 py-1 text-sm" />
        </div>
        <button onClick={grantConsent} className="px-4 py-1 bg-blue-600 text-white rounded text-sm">Grant</button>
        {grantResult && <pre className="text-xs bg-green-50 rounded p-2">{JSON.stringify(grantResult, null, 2)}</pre>}
      </div>

      {/* Check */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Check Consent</h3>
        <div className="flex gap-2">
          <input type="text" placeholder="Subject ID" value={checkSubject} onChange={(e) => setCheckSubject(e.target.value)} className="flex-1 border rounded px-3 py-1 text-sm" />
          <input type="text" placeholder="Purpose" value={checkPurpose} onChange={(e) => setCheckPurpose(e.target.value)} className="border rounded px-3 py-1 text-sm w-40" />
          <button onClick={checkConsent} className="px-4 py-1 bg-green-600 text-white rounded text-sm">Check</button>
        </div>
        {checkResult && (
          <div className={`rounded p-2 text-sm ${checkResult.allowed ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"}`}>
            {checkResult.allowed ? "Allowed" : "Denied"} — Active: {String(checkResult.active)}
          </div>
        )}
      </div>

      {/* History */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Consent History</h3>
        <div className="flex gap-2">
          <input type="text" placeholder="Subject ID" value={historySubject} onChange={(e) => setHistorySubject(e.target.value)} className="flex-1 border rounded px-3 py-1 text-sm" />
          <button onClick={fetchHistory} className="px-4 py-1 bg-gray-600 text-white rounded text-sm">Lookup</button>
        </div>
        {history && (
          <div className="space-y-1">
            {history.records.length === 0 ? (
              <p className="text-gray-400 text-sm">No records found</p>
            ) : (
              history.records.map((r, i) => (
                <div key={i} className="bg-gray-50 rounded p-2 text-xs">
                  <span className={`font-medium ${r.action === "grant" ? "text-green-600" : "text-red-600"}`}>{String(r.action).toUpperCase()}</span>
                  {" — "}{String(r.purpose)} at {String(r.timestamp)}
                  {r.reason ? <span className="text-gray-400"> ({String(r.reason)})</span> : null}
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Revoke */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Revoke Consent</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <input type="text" placeholder="Subject ID" value={revokeSubject} onChange={(e) => setRevokeSubject(e.target.value)} className="border rounded px-3 py-1 text-sm" />
          <input type="text" placeholder="Purpose" value={revokePurpose} onChange={(e) => setRevokePurpose(e.target.value)} className="border rounded px-3 py-1 text-sm" />
          <input type="text" placeholder="Reason" value={revokeReason} onChange={(e) => setRevokeReason(e.target.value)} className="border rounded px-3 py-1 text-sm" />
        </div>
        <button onClick={handleRevokeClick} className="px-4 py-1 bg-red-600 text-white rounded text-sm">Revoke</button>
        {revokeResult && <pre className="text-xs bg-red-50 rounded p-2">{JSON.stringify(revokeResult, null, 2)}</pre>}
      </div>

      {/* Confirmation Dialog for Revoke */}
      <ConfirmDialog
        open={confirmOpen}
        title="Revoke Consent"
        message={`This will permanently revoke consent for subject "${revokeSubject}" (purpose: ${revokePurpose}). This action cannot be undone.`}
        confirmLabel="Revoke"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={executeRevoke}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  );
}
