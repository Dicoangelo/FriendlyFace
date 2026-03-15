import { useState, useEffect, useCallback } from "react";
import { useToast } from "../hooks/useToast";
import { SkeletonTable } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";
import ConfirmDialog from "../components/ConfirmDialog";
import LoadingButton from "../components/LoadingButton";

interface ConsentRecord {
  id: string;
  subject_id: string;
  purpose: string;
  granted: boolean;
  timestamp: string;
  expiry: string | null;
  revocation_reason: string | null;
  event_id: string | null;
}

interface HistoryRecord {
  id: string;
  subject_id: string;
  purpose: string;
  granted: boolean;
  timestamp: string;
  expiry: string | null;
  revocation_reason: string | null;
  event_id: string | null;
}

function statusLabel(rec: ConsentRecord): "granted" | "revoked" | "expired" {
  if (!rec.granted) return "revoked";
  if (rec.expiry && new Date(rec.expiry) <= new Date()) return "expired";
  return "granted";
}

function statusBadge(status: "granted" | "revoked" | "expired") {
  const cls: Record<string, string> = {
    granted: "bg-teal/10 text-teal",
    revoked: "bg-rose-ember/10 text-rose-ember",
    expired: "bg-gold/10 text-gold",
  };
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${cls[status]}`}>
      {status}
    </span>
  );
}

export default function ConsentManagement() {
  const toast = useToast();

  // Consent list
  const [consents, setConsents] = useState<ConsentRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState<"all" | "granted" | "revoked">("all");

  // Grant form
  const [showGrant, setShowGrant] = useState(false);
  const [grantSubject, setGrantSubject] = useState("");
  const [grantPurpose, setGrantPurpose] = useState("");
  const [grantExpiry, setGrantExpiry] = useState("");
  const [grantMeta, setGrantMeta] = useState("");
  const [granting, setGranting] = useState(false);

  // Revoke
  const [revokeTarget, setRevokeTarget] = useState<ConsentRecord | null>(null);
  const [revoking, setRevoking] = useState(false);

  // Audit trail expansion
  const [expandedSubject, setExpandedSubject] = useState<string | null>(null);
  const [auditTrail, setAuditTrail] = useState<HistoryRecord[]>([]);
  const [auditLoading, setAuditLoading] = useState(false);

  const fetchConsents = useCallback(() => {
    setLoading(true);
    setError("");
    const qs = filter !== "all" ? `?status=${filter}` : "";
    fetch(`/api/v1/consent/list${qs}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setConsents(data.items);
        setTotal(data.total);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [filter]);

  useEffect(() => {
    fetchConsents();
  }, [fetchConsents]);

  const grantConsent = () => {
    if (!grantSubject.trim() || !grantPurpose.trim()) return;
    setGranting(true);
    setError("");
    fetch("/api/v1/consent/grant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        subject_id: grantSubject.trim(),
        purpose: grantPurpose.trim(),
        expiry: grantExpiry || null,
        metadata: grantMeta || undefined,
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(() => {
        toast.success(`Consent granted for "${grantSubject.trim()}"`);
        setGrantSubject("");
        setGrantPurpose("");
        setGrantExpiry("");
        setGrantMeta("");
        setShowGrant(false);
        fetchConsents();
      })
      .catch((e) => {
        setError(`Failed to grant consent: ${e.message}`);
        toast.error(`Failed to grant consent: ${e.message}`);
      })
      .finally(() => setGranting(false));
  };

  const executeRevoke = useCallback(() => {
    if (!revokeTarget) return;
    setRevoking(true);
    setError("");
    fetch("/api/v1/consent/revoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        subject_id: revokeTarget.subject_id,
        purpose: revokeTarget.purpose,
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(() => {
        toast.warning(`Consent revoked for "${revokeTarget.subject_id}"`);
        setRevokeTarget(null);
        fetchConsents();
      })
      .catch((e) => {
        setError(`Failed to revoke: ${e.message}`);
        toast.error(`Failed to revoke: ${e.message}`);
      })
      .finally(() => setRevoking(false));
  }, [revokeTarget, fetchConsents, toast]);

  const toggleAuditTrail = (subjectId: string) => {
    if (expandedSubject === subjectId) {
      setExpandedSubject(null);
      setAuditTrail([]);
      return;
    }
    setExpandedSubject(subjectId);
    setAuditLoading(true);
    fetch(`/api/v1/consent/history/${encodeURIComponent(subjectId)}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => setAuditTrail(data.records ?? []))
      .catch((e) => {
        setError(`Failed to load audit trail: ${e.message}`);
        setExpandedSubject(null);
      })
      .finally(() => setAuditLoading(false));
  };

  const filters: Array<{ label: string; value: "all" | "granted" | "revoked" }> = [
    { label: "All", value: "all" },
    { label: "Granted", value: "granted" },
    { label: "Revoked", value: "revoked" },
  ];

  return (
    <div className="space-y-6">
      {/* Error banner */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError("")} className="ml-2 font-bold hover:opacity-70">&times;</button>
        </div>
      )}

      {/* Header: filter + grant button */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="inline-flex bg-surface rounded-full p-1">
          {filters.map((f) => (
            <button
              key={f.value}
              onClick={() => setFilter(f.value)}
              className={`px-3 py-1 text-sm rounded-full transition-colors ${
                filter === f.value ? "bg-amethyst text-white" : "text-fg-secondary hover:text-fg"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <button onClick={() => setShowGrant(!showGrant)} className="btn-primary text-sm">
          {showGrant ? "Cancel" : "Grant Consent"}
        </button>
      </div>

      {/* Grant form */}
      {showGrant && (
        <div className="glass-card p-4 space-y-3 animate-fade-in">
          <h3 className="font-semibold text-fg-secondary">Grant Consent</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <input
              type="text"
              placeholder="Subject ID *"
              value={grantSubject}
              onChange={(e) => setGrantSubject(e.target.value)}
              className="ff-input"
            />
            <input
              type="text"
              placeholder="Purpose *"
              value={grantPurpose}
              onChange={(e) => setGrantPurpose(e.target.value)}
              className="ff-input"
            />
            <input
              type="date"
              value={grantExpiry}
              onChange={(e) => setGrantExpiry(e.target.value)}
              className="ff-input"
              title="Expiry date (optional)"
            />
            <input
              type="text"
              placeholder="Metadata (optional)"
              value={grantMeta}
              onChange={(e) => setGrantMeta(e.target.value)}
              className="ff-input"
            />
          </div>
          <LoadingButton
            onClick={grantConsent}
            loading={granting}
            loadingText="Granting..."
            disabled={!grantSubject.trim() || !grantPurpose.trim()}
          >
            Grant Consent
          </LoadingButton>
        </div>
      )}

      {/* Consent list */}
      {loading ? (
        <SkeletonTable rows={5} />
      ) : consents.length === 0 ? (
        <EmptyState
          title="No consent records"
          subtitle={filter !== "all" ? `No ${filter} consents found. Try a different filter.` : "Grant consent to get started."}
          action={
            filter === "all" ? (
              <button onClick={() => setShowGrant(true)} className="btn-primary text-sm">
                Grant your first consent
              </button>
            ) : undefined
          }
        />
      ) : (
        <div className="glass-card overflow-hidden">
          <div className="px-4 py-2 border-b border-border-theme flex items-center justify-between">
            <span className="text-sm text-fg-faint">{total} consent{total !== 1 ? "s" : ""}</span>
          </div>
          <div className="divide-y divide-border-theme">
            {consents.map((c) => {
              const st = statusLabel(c);
              const isExpanded = expandedSubject === c.subject_id;
              return (
                <div key={c.id}>
                  <div className="flex items-center gap-3 px-4 py-3 hover:bg-surface/50 transition-colors">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-mono text-sm truncate max-w-[200px]" title={c.subject_id}>
                          {c.subject_id}
                        </span>
                        <span className="text-fg-faint text-xs">{c.purpose}</span>
                        {statusBadge(st)}
                      </div>
                      <div className="text-xs text-fg-faint mt-0.5">
                        Granted: {new Date(c.timestamp).toLocaleString()}
                        {c.expiry && <span className="ml-2">Expires: {new Date(c.expiry).toLocaleDateString()}</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        onClick={() => toggleAuditTrail(c.subject_id)}
                        className="btn-ghost text-xs"
                      >
                        {isExpanded ? "Hide trail" : "Audit trail"}
                      </button>
                      {st === "granted" && (
                        <button
                          onClick={() => setRevokeTarget(c)}
                          className="text-xs px-2 py-1 rounded bg-rose-ember/10 text-rose-ember hover:bg-rose-ember/20 transition-colors"
                        >
                          Revoke
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Expanded audit trail */}
                  {isExpanded && (
                    <div className="px-6 pb-3 animate-fade-in">
                      {auditLoading ? (
                        <div className="py-2 text-sm text-fg-faint animate-pulse">Loading audit trail...</div>
                      ) : auditTrail.length === 0 ? (
                        <div className="py-2 text-sm text-fg-faint">No audit events found</div>
                      ) : (
                        <div className="space-y-1 border-l-2 border-border-theme pl-3">
                          {auditTrail.map((h) => (
                            <div key={h.id} className="flex items-center gap-2 text-xs py-1">
                              <span
                                className={`font-medium ${h.granted ? "text-teal" : "text-rose-ember"}`}
                              >
                                {h.granted ? "GRANT" : "REVOKE"}
                              </span>
                              <span className="text-fg-faint">{h.purpose}</span>
                              <span className="text-fg-faint">{new Date(h.timestamp).toLocaleString()}</span>
                              {h.revocation_reason && (
                                <span className="text-fg-faint italic">({h.revocation_reason})</span>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Revoke confirmation dialog */}
      <ConfirmDialog
        open={!!revokeTarget}
        title="Revoke Consent"
        message={`This will revoke consent for subject "${revokeTarget?.subject_id}" (purpose: ${revokeTarget?.purpose}). This action cannot be undone.`}
        confirmLabel={revoking ? "Revoking..." : "Revoke"}
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={executeRevoke}
        onCancel={() => setRevokeTarget(null)}
      />
    </div>
  );
}
