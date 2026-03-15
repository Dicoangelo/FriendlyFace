import { useEffect, useState } from "react";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import { SkeletonRow } from "../components/Skeleton";

interface ErasureRequest {
  request_id: string;
  subject_id: string;
  erasure_event_id: string;
  timestamp: string;
  reason?: string;
}

export default function DataErasure() {
  const [requests, setRequests] = useState<ErasureRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Erase form
  const [subjectId, setSubjectId] = useState("");
  const [reason, setReason] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [erasing, setErasing] = useState(false);
  const [eraseResult, setEraseResult] = useState<ErasureRequest | null>(null);

  const fetchRequests = () => {
    setLoading(true);
    fetch("/api/v1/erasure/records")
      .then((r) => r.json())
      .then((data) => {
        setRequests(data.items || data.requests || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  useEffect(() => {
    fetchRequests();
  }, []);

  const handleErase = () => {
    if (!subjectId.trim()) {
      setError("Subject ID is required");
      return;
    }
    setError("");
    setEraseResult(null);
    setErasing(true);

    fetch(`/api/v1/erasure/erase/${encodeURIComponent(subjectId.trim())}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason: reason.trim() || undefined }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setEraseResult(data);
        setErasing(false);
        setSubjectId("");
        setReason("");
        setConfirmed(false);
        fetchRequests();
      })
      .catch((e) => {
        setError(`Erasure error: ${e.message}`);
        setErasing(false);
      });
  };

  return (
    <div className="space-y-6">
      {/* Warning banner */}
      <div className="bg-gold/10 border border-gold/20 rounded-lg px-4 py-3 text-gold text-sm flex items-start gap-2">
        <svg className="w-5 h-5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
        </svg>
        <div>
          <p className="font-medium">Forensic Chain Notice</p>
          <p className="text-xs text-gold/80 mt-0.5">
            Subject data is cryptographically erased, but the erasure event itself is recorded in the forensic chain for audit compliance.
            Data is removed; the fact that it was removed is preserved.
          </p>
        </div>
      </div>

      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Erasure request form */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Submit Erasure Request</h3>
        <p className="text-xs text-fg-muted">
          GDPR Article 17 — Right to Erasure. This action is irreversible.
        </p>
        <label className="text-sm text-fg-secondary block">
          Subject ID
          <input
            type="text"
            value={subjectId}
            onChange={(e) => setSubjectId(e.target.value)}
            className="ff-input w-full block mt-1"
            placeholder="e.g. subject_001"
          />
        </label>
        <label className="text-sm text-fg-secondary block">
          Reason
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="ff-input w-full block mt-1 min-h-[80px]"
            placeholder="Describe the legal basis for this erasure request..."
          />
        </label>
        <label className="flex items-center gap-2 text-sm text-fg-secondary cursor-pointer">
          <input
            type="checkbox"
            checked={confirmed}
            onChange={(e) => setConfirmed(e.target.checked)}
            className="rounded border-border-theme"
          />
          I confirm this erasure is legally required
        </label>
        <LoadingButton
          onClick={handleErase}
          loading={erasing}
          disabled={!confirmed || !subjectId.trim()}
          className="btn-danger"
          loadingText="Erasing..."
        >
          Submit Erasure Request
        </LoadingButton>

        {eraseResult && (
          <div className="bg-teal/10 border border-teal/20 rounded-lg p-3 text-sm animate-fade-in">
            <p className="font-semibold text-teal mb-2">Erasure Complete</p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-fg-muted">Request ID</span>
                <p className="font-mono text-fg-secondary">{eraseResult.request_id || "—"}</p>
              </div>
              <div>
                <span className="text-fg-muted">Subject ID</span>
                <p className="font-mono text-fg-secondary">{eraseResult.subject_id || "—"}</p>
              </div>
              <div>
                <span className="text-fg-muted">Event ID</span>
                <p className="font-mono text-fg-secondary">{eraseResult.erasure_event_id || "—"}</p>
              </div>
              <div>
                <span className="text-fg-muted">Timestamp</span>
                <p className="text-fg-secondary">{eraseResult.timestamp ? new Date(eraseResult.timestamp).toLocaleString() : "—"}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Erasure history */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">
          Erasure History
        </h3>
        {loading ? (
          <div className="space-y-1">
            {[...Array(3)].map((_, i) => <SkeletonRow key={i} />)}
          </div>
        ) : requests.length === 0 ? (
          <EmptyState title="No erasure requests" subtitle="Erasure requests will be logged here" />
        ) : (
          <div className="space-y-2">
            {requests.map((r) => (
              <div key={r.request_id || r.subject_id} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                <div>
                  <p className="text-sm text-fg-secondary font-mono">{r.subject_id}</p>
                  <p className="text-xs text-fg-faint font-mono">
                    Event: {(r.erasure_event_id || "—").slice(0, 16)}
                  </p>
                </div>
                <p className="text-xs text-fg-faint">{r.timestamp ? new Date(r.timestamp).toLocaleString() : "—"}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
