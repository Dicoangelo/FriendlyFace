import { useState } from "react";

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
  const [bundleId, setBundleId] = useState("");
  const [bundle, setBundle] = useState<BundleDetail | null>(null);
  const [verification, setVerification] = useState<BundleVerification | null>(null);
  const [error, setError] = useState("");

  const lookupBundle = () => {
    if (!bundleId) return;
    setError("");
    setBundle(null);
    setVerification(null);
    fetch(`/api/v1/bundles/${bundleId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Bundle not found (${r.status})`);
        return r.json();
      })
      .then(setBundle)
      .catch((e) => setError(e.message));
  };

  const verifyBundle = () => {
    if (!bundleId) return;
    fetch(`/api/v1/verify/${bundleId}`, { method: "POST" })
      .then((r) => r.json())
      .then(setVerification)
      .catch((e) => setError(e.message));
  };

  const exportBundle = () => {
    if (!bundleId) return;
    fetch(`/api/v1/bundles/${bundleId}/export`)
      .then((r) => r.json())
      .then((data) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/ld+json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `bundle-${bundleId.slice(0, 8)}.jsonld`;
        a.click();
        URL.revokeObjectURL(url);
      })
      .catch((e) => setError(e.message));
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-fg">Bundle Inspector</h2>

      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Enter bundle ID..."
          value={bundleId}
          onChange={(e) => setBundleId(e.target.value)}
          className="flex-1 ff-input font-mono"
        />
        <button onClick={lookupBundle} className="btn-primary">
          Lookup
        </button>
      </div>

      {error && <div className="text-rose-ember text-sm">{error}</div>}

      {bundle && (
        <div className="glass-card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-fg-secondary">Bundle: {bundle.id.slice(0, 12)}...</h3>
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
            <p><span className="text-fg-muted">Status:</span> {bundle.status}</p>
            <p><span className="text-fg-muted">Events:</span> {bundle.event_ids.length}</p>
            <p><span className="text-fg-muted">Created:</span> {new Date(bundle.created_at).toLocaleString()}</p>
            <p className="col-span-2"><span className="text-fg-muted">Hash:</span> <code className="text-xs">{bundle.bundle_hash}</code></p>
            <p className="col-span-2"><span className="text-fg-muted">Merkle Root:</span> <code className="text-xs">{bundle.merkle_root}</code></p>
          </div>

          {bundle.zk_proof_placeholder && (
            <div>
              <h4 className="text-sm font-medium text-fg-secondary">ZK Proof</h4>
              <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">{bundle.zk_proof_placeholder}</pre>
            </div>
          )}

          {bundle.did_credential_placeholder && (
            <div>
              <h4 className="text-sm font-medium text-fg-secondary">DID Credential</h4>
              <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">{bundle.did_credential_placeholder}</pre>
            </div>
          )}
        </div>
      )}

      {verification && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-2">Verification Results</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {Object.entries(verification).map(([key, val]) => (
              <div key={key} className={`rounded p-2 text-center text-sm ${val ? "bg-teal/10 text-teal" : "bg-rose-ember/10 text-rose-ember"}`}>
                <span className="block font-medium">{val ? "\u2713" : "\u2717"}</span>
                {key.replace(/_/g, " ")}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
