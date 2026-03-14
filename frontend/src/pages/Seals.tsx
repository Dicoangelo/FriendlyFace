import { useEffect, useState, useCallback } from "react";
import { SkeletonCard } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import { useCopyToClipboard } from "../hooks/useCopyToClipboard";

/* ---------- types ---------- */

interface Seal {
  seal_id: string;
  system_name: string;
  issued_at: string;
  expires_at: string;
  status: string;
  compliance_score: number;
  layer_scores?: Record<string, number>;
  evidence_links?: string[];
  days_remaining?: number;
}

interface SealStatus {
  seal_id: string;
  status: string;
  days_remaining: number;
  compliance_score: number;
  layer_scores?: Record<string, number>;
  evidence_links?: string[];
}

/* ---------- helpers ---------- */

function statusColor(status: string, daysRemaining?: number): { bg: string; text: string; label: string } {
  if (status === "revoked" || status === "expired") {
    return { bg: "bg-rose-ember/10", text: "text-rose-ember", label: status };
  }
  if (status === "active" && daysRemaining !== undefined && daysRemaining <= 30) {
    return { bg: "bg-gold/10", text: "text-gold", label: "expiring" };
  }
  if (status === "active") {
    return { bg: "bg-teal/10", text: "text-teal", label: "active" };
  }
  return { bg: "bg-fg/5", text: "text-fg-secondary", label: status };
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function truncateId(id: string): string {
  if (id.length <= 16) return id;
  return id.slice(0, 8) + "..." + id.slice(-4);
}

/* ---------- main component ---------- */

export default function Seals() {
  const copy = useCopyToClipboard();

  // List state
  const [seals, setSeals] = useState<Seal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  // Detail / expand
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [sealDetail, setSealDetail] = useState<SealStatus | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Issue form
  const [showIssue, setShowIssue] = useState(false);
  const [issueSystemId, setIssueSystemId] = useState("");
  const [issueSystemName, setIssueSystemName] = useState("");
  const [issueScope, setIssueScope] = useState("");
  const [issueBundleIds, setIssueBundleIds] = useState("");
  const [issuing, setIssuing] = useState(false);

  // Verify form
  const [verifyJson, setVerifyJson] = useState("");
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<Record<string, unknown> | null>(null);

  // Revoke dialog
  const [revokeId, setRevokeId] = useState<string | null>(null);
  const [revokeReason, setRevokeReason] = useState("");
  const [revoking, setRevoking] = useState(false);

  // Renew
  const [renewingId, setRenewingId] = useState<string | null>(null);

  /* ---------- fetch seals ---------- */

  const fetchSeals = useCallback(() => {
    setLoading(true);
    setError("");
    fetch("/api/v1/seals")
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load seals (${r.status})`);
        return r.json();
      })
      .then((data) => {
        const items = Array.isArray(data) ? data : data.items || [];
        setSeals(items);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    fetchSeals();
  }, [fetchSeals]);

  /* ---------- expand row ---------- */

  const toggleExpand = (sealId: string) => {
    if (expandedId === sealId) {
      setExpandedId(null);
      setSealDetail(null);
      return;
    }
    setExpandedId(sealId);
    setSealDetail(null);
    setDetailLoading(true);
    fetch(`/api/v1/seal/status/${sealId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load seal status (${r.status})`);
        return r.json();
      })
      .then(setSealDetail)
      .catch((e) => setError(e.message))
      .finally(() => setDetailLoading(false));
  };

  /* ---------- issue new seal ---------- */

  const handleIssue = () => {
    if (!issueSystemId.trim() || !issueSystemName.trim()) {
      setError("System ID and System Name are required");
      return;
    }
    const bundleIds = issueBundleIds
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter(Boolean);

    setIssuing(true);
    setError("");
    setSuccessMsg("");
    fetch("/api/v1/seal/issue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_id: issueSystemId.trim(),
        system_name: issueSystemName.trim(),
        assessment_scope: issueScope.trim() || undefined,
        bundle_ids: bundleIds.length > 0 ? bundleIds : undefined,
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`Issue failed (${r.status})`);
        return r.json();
      })
      .then((data) => {
        setSuccessMsg(`Seal issued: ${data.seal_id || data.id || "success"}`);
        setShowIssue(false);
        setIssueSystemId("");
        setIssueSystemName("");
        setIssueScope("");
        setIssueBundleIds("");
        fetchSeals();
      })
      .catch((e) => setError(e.message))
      .finally(() => setIssuing(false));
  };

  /* ---------- verify ---------- */

  const handleVerify = () => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(verifyJson);
    } catch {
      setError("Invalid JSON — please paste a valid credential");
      return;
    }
    setVerifying(true);
    setError("");
    setVerifyResult(null);
    fetch("/api/v1/seal/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential: parsed }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`Verify failed (${r.status})`);
        return r.json();
      })
      .then((data) => setVerifyResult(data as Record<string, unknown>))
      .catch((e) => setError(e.message))
      .finally(() => setVerifying(false));
  };

  /* ---------- renew ---------- */

  const handleRenew = (sealId: string) => {
    setRenewingId(sealId);
    setError("");
    fetch(`/api/v1/seal/renew/${sealId}`, { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`Renew failed (${r.status})`);
        return r.json();
      })
      .then(() => {
        setSuccessMsg(`Seal ${truncateId(sealId)} renewed`);
        fetchSeals();
        if (expandedId === sealId) toggleExpand(sealId);
      })
      .catch((e) => setError(e.message))
      .finally(() => setRenewingId(null));
  };

  /* ---------- revoke ---------- */

  const handleRevoke = () => {
    if (!revokeId) return;
    setRevoking(true);
    setError("");
    fetch(`/api/v1/seal/revoke/${revokeId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason: revokeReason.trim() || "Manual revocation" }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`Revoke failed (${r.status})`);
        return r.json();
      })
      .then(() => {
        setSuccessMsg(`Seal ${truncateId(revokeId)} revoked`);
        setRevokeId(null);
        setRevokeReason("");
        fetchSeals();
        if (expandedId === revokeId) {
          setExpandedId(null);
          setSealDetail(null);
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setRevoking(false));
  };

  /* ---------- render ---------- */

  if (loading) {
    return (
      <div className="space-y-4">
        <SkeletonCard className="h-24" />
        <SkeletonCard className="h-24" />
        <SkeletonCard className="h-24" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Error banner */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/30 text-rose-ember rounded-lg px-4 py-3 text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError("")} className="text-rose-ember/70 hover:text-rose-ember ml-4">
            &times;
          </button>
        </div>
      )}

      {/* Success banner */}
      {successMsg && (
        <div className="bg-teal/10 border border-teal/30 text-teal rounded-lg px-4 py-3 text-sm flex items-center justify-between">
          <span>{successMsg}</span>
          <button onClick={() => setSuccessMsg("")} className="text-teal/70 hover:text-teal ml-4">
            &times;
          </button>
        </div>
      )}

      {/* Header row */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="text-sm text-fg-muted font-medium">{seals.length} seal{seals.length !== 1 ? "s" : ""}</span>
        <button onClick={() => setShowIssue(!showIssue)} className="btn-primary">
          {showIssue ? "Cancel" : "Issue New Seal"}
        </button>
      </div>

      {/* Issue new seal form */}
      {showIssue && (
        <div className="glass-card p-4 space-y-3 animate-fade-in">
          <h3 className="text-sm font-semibold text-fg-secondary">Issue New Seal</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-fg-muted mb-1">System ID *</label>
              <input
                type="text"
                value={issueSystemId}
                onChange={(e) => setIssueSystemId(e.target.value)}
                className="w-full bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40"
                placeholder="e.g. sys-001"
              />
            </div>
            <div>
              <label className="block text-xs text-fg-muted mb-1">System Name *</label>
              <input
                type="text"
                value={issueSystemName}
                onChange={(e) => setIssueSystemName(e.target.value)}
                className="w-full bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40"
                placeholder="e.g. FaceRecognition-Prod"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs text-fg-muted mb-1">Assessment Scope (optional)</label>
            <input
              type="text"
              value={issueScope}
              onChange={(e) => setIssueScope(e.target.value)}
              className="w-full bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40"
              placeholder="e.g. full-audit, bias-only"
            />
          </div>
          <div>
            <label className="block text-xs text-fg-muted mb-1">Bundle IDs (optional, comma-separated)</label>
            <textarea
              value={issueBundleIds}
              onChange={(e) => setIssueBundleIds(e.target.value)}
              rows={2}
              className="w-full bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm font-mono text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40"
              placeholder="bundle-id-1, bundle-id-2"
            />
          </div>
          <LoadingButton onClick={handleIssue} loading={issuing} loadingText="Issuing...">
            Issue Seal
          </LoadingButton>
        </div>
      )}

      {/* Verify external seal */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="text-sm font-semibold text-fg-secondary">Verify External Seal</h3>
        <div className="flex flex-col sm:flex-row gap-2">
          <textarea
            value={verifyJson}
            onChange={(e) => setVerifyJson(e.target.value)}
            rows={2}
            className="flex-1 bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm font-mono text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40"
            placeholder='Paste credential JSON here...'
          />
          <LoadingButton
            onClick={handleVerify}
            loading={verifying}
            loadingText="Verifying..."
            disabled={!verifyJson.trim()}
            className="btn-accent self-start"
          >
            Verify
          </LoadingButton>
        </div>
        {verifyResult && (
          <div className="animate-fade-in">
            <pre className="bg-surface rounded-lg p-3 text-xs font-mono text-fg-secondary max-h-48 overflow-auto whitespace-pre-wrap">
              {JSON.stringify(verifyResult, null, 2)}
            </pre>
          </div>
        )}
      </div>

      {/* Revoke dialog */}
      {revokeId && (
        <div className="glass-card p-4 space-y-3 animate-fade-in border border-rose-ember/30">
          <h3 className="text-sm font-semibold text-rose-ember">Revoke Seal: {truncateId(revokeId)}</h3>
          <div>
            <label className="block text-xs text-fg-muted mb-1">Reason</label>
            <input
              type="text"
              value={revokeReason}
              onChange={(e) => setRevokeReason(e.target.value)}
              className="w-full bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-rose-ember/40"
              placeholder="Reason for revocation..."
            />
          </div>
          <div className="flex gap-2">
            <LoadingButton onClick={handleRevoke} loading={revoking} loadingText="Revoking..." className="btn-primary bg-rose-ember/20 text-rose-ember hover:bg-rose-ember/30">
              Confirm Revoke
            </LoadingButton>
            <button onClick={() => { setRevokeId(null); setRevokeReason(""); }} className="text-sm text-fg-muted hover:text-fg-secondary">
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Seal list or empty state */}
      {seals.length === 0 ? (
        <div className="glass-card">
          <EmptyState
            title="No seals issued yet"
            subtitle="Issue a ForensicSeal to certify system compliance and integrity"
            action={
              <button onClick={() => setShowIssue(true)} className="btn-primary">
                Issue your first seal
              </button>
            }
          />
        </div>
      ) : (
        <>
          {/* Compliance history chart */}
          <ComplianceChart seals={seals} />

          {/* Table */}
          <div className="glass-card overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border-theme text-left">
                  <th className="px-4 py-3 text-xs font-semibold text-fg-muted uppercase tracking-wider">Seal ID</th>
                  <th className="px-4 py-3 text-xs font-semibold text-fg-muted uppercase tracking-wider">System</th>
                  <th className="px-4 py-3 text-xs font-semibold text-fg-muted uppercase tracking-wider hidden sm:table-cell">Issued</th>
                  <th className="px-4 py-3 text-xs font-semibold text-fg-muted uppercase tracking-wider hidden md:table-cell">Expires</th>
                  <th className="px-4 py-3 text-xs font-semibold text-fg-muted uppercase tracking-wider">Status</th>
                  <th className="px-4 py-3 text-xs font-semibold text-fg-muted uppercase tracking-wider text-right">Score</th>
                </tr>
              </thead>
              <tbody>
                {seals.map((seal) => {
                  const sc = statusColor(seal.status, seal.days_remaining);
                  const isExpanded = expandedId === seal.seal_id;
                  return (
                    <tbody key={seal.seal_id}>
                      <tr
                        className="border-b border-border-theme/50 hover:bg-fg/5 cursor-pointer transition-colors"
                        onClick={() => toggleExpand(seal.seal_id)}
                      >
                        <td className="px-4 py-3">
                          <button
                            onClick={(e) => { e.stopPropagation(); copy(seal.seal_id, "Seal ID copied"); }}
                            className="font-mono text-xs text-fg-secondary hover:text-cyan transition-colors"
                            title="Click to copy full ID"
                          >
                            {truncateId(seal.seal_id)}
                          </button>
                        </td>
                        <td className="px-4 py-3 text-fg-secondary">{seal.system_name}</td>
                        <td className="px-4 py-3 text-fg-faint hidden sm:table-cell">{formatDate(seal.issued_at)}</td>
                        <td className="px-4 py-3 text-fg-faint hidden md:table-cell">{formatDate(seal.expires_at)}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium ${sc.bg} ${sc.text}`}>
                            {sc.label}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <span className="font-mono text-xs text-fg-secondary">
                            {(seal.compliance_score * 100).toFixed(1)}%
                          </span>
                        </td>
                      </tr>

                      {/* Expanded detail row */}
                      {isExpanded && (
                        <tr>
                          <td colSpan={6} className="px-4 py-4 bg-surface/50">
                            <div className="animate-fade-in space-y-4">
                              {detailLoading ? (
                                <div className="text-xs text-fg-faint py-4 text-center">Loading details...</div>
                              ) : sealDetail ? (
                                <>
                                  {/* Status summary */}
                                  <div className="flex flex-wrap gap-4 text-xs">
                                    <div>
                                      <span className="text-fg-faint">Days remaining: </span>
                                      <span className={`font-medium ${sealDetail.days_remaining <= 30 ? "text-gold" : "text-teal"}`}>
                                        {sealDetail.days_remaining}
                                      </span>
                                    </div>
                                    <div>
                                      <span className="text-fg-faint">Overall score: </span>
                                      <span className="font-medium text-fg-secondary">
                                        {(sealDetail.compliance_score * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                  </div>

                                  {/* Per-layer scores */}
                                  {sealDetail.layer_scores && Object.keys(sealDetail.layer_scores).length > 0 && (
                                    <div>
                                      <h4 className="text-xs font-semibold text-fg-muted mb-2 uppercase tracking-wider">Per-Layer Scores</h4>
                                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                                        {Object.entries(sealDetail.layer_scores).map(([layer, score]) => (
                                          <div key={layer} className="flex items-center gap-2">
                                            <span className="text-xs text-fg-faint w-28 truncate" title={layer}>{layer}</span>
                                            <div className="flex-1 h-2 bg-fg/10 rounded-full overflow-hidden">
                                              <div
                                                className="h-full rounded-full transition-all"
                                                style={{
                                                  width: `${Math.min(score * 100, 100)}%`,
                                                  backgroundColor: score >= 0.8 ? "var(--color-teal, #2dd4bf)" : score >= 0.5 ? "var(--color-gold, #f59e0b)" : "var(--color-rose-ember, #f43f5e)",
                                                }}
                                              />
                                            </div>
                                            <span className="text-xs font-mono text-fg-secondary w-12 text-right">
                                              {(score * 100).toFixed(0)}%
                                            </span>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}

                                  {/* Evidence links */}
                                  {sealDetail.evidence_links && sealDetail.evidence_links.length > 0 && (
                                    <div>
                                      <h4 className="text-xs font-semibold text-fg-muted mb-2 uppercase tracking-wider">Evidence</h4>
                                      <div className="space-y-1">
                                        {sealDetail.evidence_links.map((link, i) => (
                                          <a
                                            key={i}
                                            href={link}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="block text-xs text-cyan hover:text-cyan-dim truncate"
                                          >
                                            {link}
                                          </a>
                                        ))}
                                      </div>
                                    </div>
                                  )}

                                  {/* Actions */}
                                  <div className="flex gap-2 pt-2 border-t border-border-theme">
                                    {seal.status === "active" && (
                                      <>
                                        <LoadingButton
                                          onClick={() => handleRenew(seal.seal_id)}
                                          loading={renewingId === seal.seal_id}
                                          loadingText="Renewing..."
                                          className="btn-accent text-xs px-3 py-1"
                                        >
                                          Renew
                                        </LoadingButton>
                                        <button
                                          onClick={() => setRevokeId(seal.seal_id)}
                                          className="text-xs px-3 py-1 rounded-lg text-rose-ember hover:bg-rose-ember/10 transition-colors"
                                        >
                                          Revoke
                                        </button>
                                      </>
                                    )}
                                    <button
                                      onClick={() => copy(JSON.stringify(sealDetail, null, 2), "Seal details copied")}
                                      className="text-xs px-3 py-1 rounded-lg text-fg-muted hover:text-fg-secondary hover:bg-fg/5 transition-colors"
                                    >
                                      Copy JSON
                                    </button>
                                  </div>
                                </>
                              ) : (
                                <div className="text-xs text-fg-faint py-2">Failed to load seal details</div>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

/* ---------- compliance history chart (inline CSS bars) ---------- */

function ComplianceChart({ seals }: { seals: Seal[] }) {
  // Sort by issued_at ascending, show last 20
  const sorted = [...seals]
    .sort((a, b) => new Date(a.issued_at).getTime() - new Date(b.issued_at).getTime())
    .slice(-20);

  if (sorted.length < 2) return null;

  const maxScore = Math.max(...sorted.map((s) => s.compliance_score), 1);

  return (
    <div className="glass-card p-4">
      <h3 className="text-xs font-semibold text-fg-muted uppercase tracking-wider mb-3">Compliance Score Over Time</h3>
      <div className="flex items-end gap-1" style={{ height: "120px" }}>
        {sorted.map((seal) => {
          const heightPct = (seal.compliance_score / maxScore) * 100;
          const sc = statusColor(seal.status, seal.days_remaining);
          const barColor =
            seal.compliance_score >= 0.8
              ? "var(--color-teal, #2dd4bf)"
              : seal.compliance_score >= 0.5
                ? "var(--color-gold, #f59e0b)"
                : "var(--color-rose-ember, #f43f5e)";

          return (
            <div
              key={seal.seal_id}
              className="flex-1 min-w-[8px] max-w-[40px] group relative"
              style={{ height: "100%" }}
            >
              <div className="absolute bottom-0 w-full rounded-t transition-all hover:opacity-80"
                style={{
                  height: `${heightPct}%`,
                  backgroundColor: barColor,
                  minHeight: "4px",
                }}
              />
              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10">
                <div className="bg-surface border border-border-theme rounded-lg px-2 py-1 text-xs whitespace-nowrap shadow-lg">
                  <div className="text-fg-secondary font-medium">{seal.system_name}</div>
                  <div className="text-fg-faint">{formatDate(seal.issued_at)}</div>
                  <div className="text-fg-faint">
                    Score: {(seal.compliance_score * 100).toFixed(1)}%
                    <span className={`ml-1 ${sc.text}`}>({sc.label})</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between mt-2 text-[10px] text-fg-faint">
        <span>{formatDate(sorted[0].issued_at)}</span>
        <span>{formatDate(sorted[sorted.length - 1].issued_at)}</span>
      </div>
    </div>
  );
}
