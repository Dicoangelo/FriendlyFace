import { useEffect, useState } from "react";
import { SkeletonTable } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";

interface BundleSummary {
  id: string;
  created_at: string;
  status: string;
  bundle_hash: string;
  merkle_root: string;
  event_count: number;
}

interface BundleVerification {
  hash_valid: boolean;
  merkle_valid: boolean;
  provenance_valid: boolean;
  zk_valid: boolean;
  did_valid: boolean;
}

interface BundleDetail {
  id: string;
  created_at: string;
  status: string;
  bundle_hash: string;
  merkle_root: string;
  event_ids: string[];
  zk_proof_placeholder?: string;
  did_credential_placeholder?: string;
}

export default function Bundles() {
  const [bundles, setBundles] = useState<BundleSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<BundleDetail | null>(null);
  const [verification, setVerification] = useState<BundleVerification | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/v1/bundles?limit=50")
      .then((r) => r.json())
      .then((data) => {
        setBundles(data.items || []);
        setTotal(data.total || 0);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const selectBundle = (id: string) => {
    setSelectedId(id);
    setVerification(null);
    setError("");
    fetch(`/api/v1/bundles/${id}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Bundle not found (${r.status})`);
        return r.json();
      })
      .then(setDetail)
      .catch((e) => setError(e.message));
  };

  const verifyBundle = () => {
    if (!selectedId) return;
    fetch(`/api/v1/verify/${selectedId}`, { method: "POST" })
      .then((r) => r.json())
      .then(setVerification)
      .catch((e) => setError(e.message));
  };

  const exportBundle = () => {
    if (!selectedId) return;
    fetch(`/api/v1/bundles/${selectedId}/export`)
      .then((r) => r.json())
      .then((data) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/ld+json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `bundle-${selectedId.slice(0, 8)}.jsonld`;
        a.click();
        URL.revokeObjectURL(url);
      })
      .catch((e) => setError(e.message));
  };

  if (loading) return <SkeletonTable rows={6} />;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm text-fg-muted font-medium">{total} bundles</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Bundle list */}
        <div className="lg:col-span-1 space-y-2">
          {bundles.length === 0 ? (
            <div className="glass-card">
              <EmptyState title="No bundles created yet" subtitle="Create a forensic bundle via the API to see it here" />
            </div>
          ) : (
            bundles.map((b) => (
              <button
                key={b.id}
                onClick={() => selectBundle(b.id)}
                className={`w-full text-left glass-card p-3 transition-all hover:scale-[1.01] ${
                  selectedId === b.id
                    ? "border-cyan/40 bg-cyan/5"
                    : "hover:border-fg-faint/30"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm text-fg-secondary">
                    {b.id.slice(0, 12)}...
                  </span>
                  <StatusBadge status={b.status} />
                </div>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-xs text-fg-faint">
                    {new Date(b.created_at).toLocaleString()}
                  </span>
                  <span className="text-xs text-fg-muted">
                    {b.event_count} event{b.event_count !== 1 ? "s" : ""}
                  </span>
                </div>
              </button>
            ))
          )}
        </div>

        {/* Bundle detail panel */}
        <div className="lg:col-span-2">
          {error && <div className="text-rose-ember text-sm mb-2">{error}</div>}

          {!selectedId && (
            <div className="glass-card p-12 text-center text-fg-faint">
              Select a bundle to inspect
            </div>
          )}

          {detail && selectedId && (
            <div className="space-y-4 animate-fade-in">
              <div className="glass-card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-fg-secondary">
                    Bundle: {detail.id.slice(0, 12)}...
                  </h3>
                  <div className="flex gap-2">
                    <button onClick={verifyBundle} className="btn-success">
                      Verify
                    </button>
                    <button onClick={exportBundle} className="btn-accent">
                      Export JSON-LD
                    </button>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <p>
                    <span className="text-fg-muted">Status:</span>{" "}
                    <StatusBadge status={detail.status} />
                  </p>
                  <p>
                    <span className="text-fg-muted">Events:</span> {detail.event_ids.length}
                  </p>
                  <p>
                    <span className="text-fg-muted">Created:</span>{" "}
                    {new Date(detail.created_at).toLocaleString()}
                  </p>
                  <p className="col-span-2">
                    <span className="text-fg-muted">Hash:</span>{" "}
                    <code className="text-xs">{detail.bundle_hash}</code>
                  </p>
                  <p className="col-span-2">
                    <span className="text-fg-muted">Merkle Root:</span>{" "}
                    <code className="text-xs">{detail.merkle_root}</code>
                  </p>
                </div>

                {detail.event_ids.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-fg-secondary mb-1">Event IDs</h4>
                    <div className="bg-surface rounded-lg p-2 max-h-32 overflow-y-auto">
                      {detail.event_ids.map((eid) => (
                        <p key={eid} className="text-xs font-mono text-fg-muted py-0.5">
                          {eid}
                        </p>
                      ))}
                    </div>
                  </div>
                )}

                {detail.zk_proof_placeholder && (
                  <div>
                    <h4 className="text-sm font-medium text-fg-secondary">ZK Proof</h4>
                    <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
                      {detail.zk_proof_placeholder}
                    </pre>
                  </div>
                )}

                {detail.did_credential_placeholder && (
                  <div>
                    <h4 className="text-sm font-medium text-fg-secondary">DID Credential</h4>
                    <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
                      {detail.did_credential_placeholder}
                    </pre>
                  </div>
                )}
              </div>

              {verification && (
                <div className="glass-card p-4">
                  <h3 className="font-semibold text-fg-secondary mb-2">Verification Results</h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                    {Object.entries(verification).map(([key, val]) => (
                      <div
                        key={key}
                        className={`rounded p-2 text-center text-sm ${
                          val
                            ? "bg-teal/10 text-teal"
                            : "bg-rose-ember/10 text-rose-ember"
                        }`}
                      >
                        <span className="block font-medium">{val ? "\u2713" : "\u2717"}</span>
                        {key.replace(/_/g, " ")}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    sealed: "bg-teal/10 text-teal",
    created: "bg-cyan/10 text-cyan",
    pending: "bg-gold/10 text-gold",
    failed: "bg-rose-ember/10 text-rose-ember",
  };
  const cls = colors[status] || "bg-fg/5 text-fg-secondary";
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {status}
    </span>
  );
}
