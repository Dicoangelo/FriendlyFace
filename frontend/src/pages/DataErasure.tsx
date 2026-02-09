import { useEffect, useState } from "react";
import EmptyState from "../components/EmptyState";

interface ErasureRecord {
  subject_id: string;
  status: string;
  erased_at: string;
  events_erased: number;
  consent_records_erased: number;
}

export default function DataErasure() {
  const [records, setRecords] = useState<ErasureRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState("");

  // Erase form
  const [eraseSubject, setEraseSubject] = useState("");
  const [erasing, setErasing] = useState(false);
  const [eraseResult, setEraseResult] = useState<Record<string, unknown> | null>(null);

  // Status check
  const [statusSubject, setStatusSubject] = useState("");
  const [statusResult, setStatusResult] = useState<Record<string, unknown> | null>(null);

  const fetchRecords = () => {
    fetch("/api/v1/erasure/records")
      .then((r) => r.json())
      .then((data) => {
        setRecords(data.items || []);
        setTotal(data.total ?? 0);
      })
      .catch(() => {});
  };

  useEffect(() => {
    fetchRecords();
  }, []);

  const handleErase = () => {
    if (!eraseSubject.trim()) {
      setError("Subject ID is required");
      return;
    }
    setError("");
    setEraseResult(null);
    setErasing(true);

    fetch(`/api/v1/erasure/erase/${encodeURIComponent(eraseSubject.trim())}`, {
      method: "POST",
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setEraseResult(data);
        setErasing(false);
        fetchRecords();
      })
      .catch((e) => {
        setError(`Erasure error: ${e.message}`);
        setErasing(false);
      });
  };

  const checkStatus = () => {
    if (!statusSubject.trim()) {
      setError("Subject ID is required");
      return;
    }
    setError("");
    setStatusResult(null);

    fetch(`/api/v1/erasure/status/${encodeURIComponent(statusSubject.trim())}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setStatusResult)
      .catch((e) => setError(`Status error: ${e.message}`));
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Erase subject */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Erase Subject Data</h3>
        <p className="text-xs text-fg-muted">
          Cryptographically erase all data for a subject (GDPR Article 17 — Right to Erasure).
          This action is irreversible and forensically logged.
        </p>
        <div className="flex gap-3 items-end">
          <label className="text-sm text-fg-secondary">
            Subject ID
            <input
              type="text"
              value={eraseSubject}
              onChange={(e) => setEraseSubject(e.target.value)}
              className="ff-input w-64 block mt-1"
              placeholder="e.g. subject_001"
            />
          </label>
          <button
            onClick={handleErase}
            disabled={erasing}
            className="px-4 py-2 rounded-lg text-sm font-medium bg-rose-ember/10 text-rose-ember border border-rose-ember/20 hover:bg-rose-ember/20 transition-colors disabled:opacity-50"
          >
            {erasing ? "Erasing..." : "Erase All Data"}
          </button>
        </div>
        {eraseResult && (
          <div className="bg-teal/10 border border-teal/20 rounded-lg px-3 py-2 text-teal text-sm">
            Erasure complete — {String(eraseResult.events_erased ?? 0)} events,{" "}
            {String(eraseResult.consent_records_erased ?? 0)} consent records removed
          </div>
        )}
      </div>

      {/* Check status */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Check Erasure Status</h3>
        <div className="flex gap-3 items-end">
          <label className="text-sm text-fg-secondary">
            Subject ID
            <input
              type="text"
              value={statusSubject}
              onChange={(e) => setStatusSubject(e.target.value)}
              className="ff-input w-64 block mt-1"
              placeholder="e.g. subject_001"
            />
          </label>
          <button onClick={checkStatus} className="btn-primary">
            Check
          </button>
        </div>
        {statusResult && (
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
            {JSON.stringify(statusResult, null, 2)}
          </pre>
        )}
      </div>

      {/* Erasure records */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">
          Erasure Records <span className="text-fg-faint font-normal">({total})</span>
        </h3>
        {records.length === 0 ? (
          <EmptyState title="No erasure records" subtitle="Subject erasures will be logged here" />
        ) : (
          <div className="space-y-2">
            {records.map((r) => (
              <div key={r.subject_id} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                <div>
                  <p className="text-sm text-fg-secondary font-mono">{r.subject_id}</p>
                  <p className="text-xs text-fg-faint">
                    {r.events_erased} events, {r.consent_records_erased} consent records
                  </p>
                </div>
                <div className="text-right">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${r.status === "erased" ? "bg-teal/10 text-teal" : "bg-gold/10 text-gold"}`}>
                    {r.status}
                  </span>
                  <p className="text-xs text-fg-faint mt-1">{new Date(r.erased_at).toLocaleString()}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
