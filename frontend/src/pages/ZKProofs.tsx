import { useState } from "react";

export default function ZKProofs() {
  const [bundleId, setBundleId] = useState("");
  const [proof, setProof] = useState<Record<string, unknown> | null>(null);
  const [verifyInput, setVerifyInput] = useState("");
  const [verifyResult, setVerifyResult] = useState<{ valid: boolean; scheme: string } | null>(null);
  const [storedProof, setStoredProof] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");

  const generateProof = () => {
    setError("");
    setProof(null);
    fetch("/api/v1/zk/prove", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bundle_id: bundleId }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setProof(data.proof);
        setVerifyInput(JSON.stringify(data.proof));
      })
      .catch((e) => setError(e.message));
  };

  const verifyProof = () => {
    setError("");
    setVerifyResult(null);
    fetch("/api/v1/zk/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ proof: verifyInput }),
    })
      .then((r) => r.json())
      .then(setVerifyResult)
      .catch((e) => setError(e.message));
  };

  const getStoredProof = () => {
    setError("");
    setStoredProof(null);
    fetch(`/api/v1/zk/proofs/${bundleId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setStoredProof)
      .catch((e) => setError(e.message));
  };

  return (
    <div className="space-y-6">
      {/* Page title shown in header bar */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Generate */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Generate Proof</h3>
        <p className="text-xs text-fg-faint">Enter a forensic bundle ID to generate a Schnorr ZK proof for chain verification.</p>
        <div className="flex gap-2">
          <input type="text" placeholder="Bundle ID (UUID)" value={bundleId} onChange={(e) => setBundleId(e.target.value)} className="flex-1 ff-input font-mono" />
          <button onClick={generateProof} disabled={!bundleId.trim()} className="btn-primary disabled:opacity-40">Prove</button>
          <button onClick={getStoredProof} disabled={!bundleId.trim()} className="btn-ghost disabled:opacity-40">Get Stored</button>
        </div>
        {proof && (
          <div className="relative">
            <pre className="text-xs bg-[rgb(var(--code-bg))] text-teal rounded p-3 overflow-x-auto">{JSON.stringify(proof, null, 2)}</pre>
            <button onClick={() => navigator.clipboard.writeText(JSON.stringify(proof, null, 2))} className="absolute top-2 right-2 text-fg-faint hover:text-fg text-xs">Copy</button>
          </div>
        )}
        {storedProof && (
          <div className="relative">
            <h4 className="text-sm font-medium text-fg-secondary">Stored Proof</h4>
            <pre className="text-xs bg-[rgb(var(--code-bg))] text-teal rounded p-3 overflow-x-auto">{JSON.stringify(storedProof, null, 2)}</pre>
          </div>
        )}
      </div>

      {/* Verify */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Verify Proof</h3>
        <p className="text-xs text-fg-faint">Paste a proof JSON to verify its cryptographic validity.</p>
        <textarea placeholder="Paste proof JSON" value={verifyInput} onChange={(e) => setVerifyInput(e.target.value)} className="w-full ff-textarea font-mono h-24" />
        <button onClick={verifyProof} disabled={!verifyInput.trim()} className="btn-success disabled:opacity-40">Verify</button>
        {verifyResult && (
          <div className={`rounded p-2 text-sm ${verifyResult.valid ? "bg-teal/10 text-teal" : "bg-rose-ember/10 text-rose-ember"}`}>
            {verifyResult.valid ? "\u2713 Valid" : "\u2717 Invalid"} — Scheme: {verifyResult.scheme}
          </div>
        )}
      </div>
    </div>
  );
}
