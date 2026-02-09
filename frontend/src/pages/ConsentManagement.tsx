import { useState, useCallback } from "react";
import { useToast } from "../hooks/useToast";
import ConfirmDialog from "../components/ConfirmDialog";
import LoadingButton from "../components/LoadingButton";

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
  const [grantLoading, setGrantLoading] = useState(false);
  const [revokeLoading, setRevokeLoading] = useState(false);

  const grantConsent = () => {
    setError("");
    setGrantResult(null);
    setGrantLoading(true);
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
      })
      .finally(() => setGrantLoading(false));
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
    setRevokeLoading(true);
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
      })
      .finally(() => setRevokeLoading(false));
  }, [revokeSubject, revokePurpose, revokeReason, toast]);

  const handleRevokeClick = () => {
    setConfirmOpen(true);
  };

  return (
    <div className="space-y-6">
      {/* Page title shown in header bar */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Grant */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Grant Consent</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <input type="text" placeholder="Subject ID" value={subjectId} onChange={(e) => setSubjectId(e.target.value)} className="ff-input" />
          <input type="text" placeholder="Purpose" value={purpose} onChange={(e) => setPurpose(e.target.value)} className="ff-input" />
          <input type="text" placeholder="Expiry (ISO-8601, optional)" value={expiry} onChange={(e) => setExpiry(e.target.value)} className="ff-input" />
        </div>
        <LoadingButton onClick={grantConsent} loading={grantLoading} loadingText="Granting...">Grant</LoadingButton>
        {grantResult && (
          <div className="flex items-center gap-2 text-teal text-sm bg-teal/10 rounded-lg px-3 py-2">
            <span className="font-bold">&#10003;</span>
            <span>Consent granted — ID: <code className="font-mono text-xs">{String(grantResult.id ?? "").slice(0, 12) || "ok"}</code></span>
          </div>
        )}
      </div>

      {/* Check */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Check Consent</h3>
        <div className="flex gap-2">
          <input type="text" placeholder="Subject ID" value={checkSubject} onChange={(e) => setCheckSubject(e.target.value)} className="flex-1 ff-input" />
          <input type="text" placeholder="Purpose" value={checkPurpose} onChange={(e) => setCheckPurpose(e.target.value)} className="ff-input w-40" />
          <button onClick={checkConsent} className="btn-success">Check</button>
        </div>
        {checkResult && (
          <div className={`rounded p-2 text-sm ${checkResult.allowed ? "bg-teal/10 text-teal" : "bg-rose-ember/10 text-rose-ember"}`}>
            {checkResult.allowed ? "Allowed" : "Denied"} — Active: {String(checkResult.active)}
          </div>
        )}
      </div>

      {/* History */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Consent History</h3>
        <div className="flex gap-2">
          <input type="text" placeholder="Subject ID" value={historySubject} onChange={(e) => setHistorySubject(e.target.value)} className="flex-1 ff-input" />
          <button onClick={fetchHistory} className="btn-ghost">Lookup</button>
        </div>
        {history && (
          <div className="space-y-1">
            {history.records.length === 0 ? (
              <p className="text-fg-faint text-sm">No records found</p>
            ) : (
              history.records.map((r, i) => (
                <div key={i} className="bg-surface rounded-lg p-2 text-xs">
                  <span className={`font-medium ${r.action === "grant" ? "text-teal" : "text-rose-ember"}`}>{String(r.action).toUpperCase()}</span>
                  {" — "}{String(r.purpose)} at {String(r.timestamp)}
                  {r.reason ? <span className="text-fg-faint"> ({String(r.reason)})</span> : null}
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Revoke */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Revoke Consent</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <input type="text" placeholder="Subject ID" value={revokeSubject} onChange={(e) => setRevokeSubject(e.target.value)} className="ff-input" />
          <input type="text" placeholder="Purpose" value={revokePurpose} onChange={(e) => setRevokePurpose(e.target.value)} className="ff-input" />
          <input type="text" placeholder="Reason" value={revokeReason} onChange={(e) => setRevokeReason(e.target.value)} className="ff-input" />
        </div>
        <LoadingButton onClick={handleRevokeClick} loading={revokeLoading} className="btn-danger" loadingText="Revoking...">Revoke</LoadingButton>
        {revokeResult && (
          <div className="flex items-center gap-2 text-rose-ember text-sm bg-rose-ember/10 rounded-lg px-3 py-2">
            <span className="font-bold">&#10003;</span>
            <span>Consent revoked successfully</span>
          </div>
        )}
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
