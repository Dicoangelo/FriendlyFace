import { useState } from "react";
import LoadingButton from "../components/LoadingButton";
import { useCopyToClipboard } from "../hooks/useCopyToClipboard";

interface ProofData {
  scheme?: string;
  commitment?: string;
  challenge?: string;
  response?: string;
  [key: string]: unknown;
}

function ProofBreakdown({ proof, label }: { proof: ProofData; label?: string }) {
  const copy = useCopyToClipboard();
  const fields: { key: keyof ProofData; label: string }[] = [
    { key: "scheme", label: "Scheme" },
    { key: "commitment", label: "Commitment" },
    { key: "challenge", label: "Challenge" },
    { key: "response", label: "Response" },
  ];

  return (
    <div className="space-y-2">
      {label && <h4 className="text-sm font-medium text-fg-secondary">{label}</h4>}
      <div className="bg-[rgb(var(--code-bg))] rounded-lg p-3 space-y-2 relative">
        <button
          onClick={() => copy(JSON.stringify(proof, null, 2), "Proof copied")}
          className="absolute top-2 right-2 text-fg-faint hover:text-fg text-xs px-2 py-1 rounded hover:bg-fg/5 transition-colors"
        >
          Copy
        </button>
        {fields.map(({ key, label }) => {
          const value = proof[key];
          if (value === undefined) return null;
          return (
            <div key={key} className="flex flex-col gap-0.5">
              <span className="text-[11px] uppercase tracking-wider text-fg-faint">{label}</span>
              <span className="text-xs font-mono text-teal break-all">{String(value)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ZKProofs() {
  const [bundleId, setBundleId] = useState("");
  const [proof, setProof] = useState<ProofData | null>(null);
  const [verifyInput, setVerifyInput] = useState("");
  const [verifyResult, setVerifyResult] = useState<{ valid: boolean; scheme: string } | null>(null);
  const [storedProof, setStoredProof] = useState<ProofData | null>(null);
  const [error, setError] = useState("");
  const [generating, setGenerating] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [fetching, setFetching] = useState(false);

  const generateProof = () => {
    setError("");
    setProof(null);
    setGenerating(true);
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
        setVerifyInput(JSON.stringify(data.proof, null, 2));
      })
      .catch((e) => setError(e.message))
      .finally(() => setGenerating(false));
  };

  const verifyProof = () => {
    setError("");
    setVerifyResult(null);
    setVerifying(true);
    let parsed;
    try {
      parsed = JSON.parse(verifyInput);
    } catch {
      setError("Invalid JSON — check your proof input");
      setVerifying(false);
      return;
    }
    fetch("/api/v1/zk/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ proof: parsed }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setVerifyResult)
      .catch((e) => setError(e.message))
      .finally(() => setVerifying(false));
  };

  const getStoredProof = () => {
    setError("");
    setStoredProof(null);
    setFetching(true);
    fetch(`/api/v1/zk/proofs/${bundleId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setStoredProof)
      .catch((e) => setError(e.message))
      .finally(() => setFetching(false));
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-3 text-rose-ember text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError("")} className="text-rose-ember/60 hover:text-rose-ember ml-4">&times;</button>
        </div>
      )}

      {/* Generate Proof */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Generate Proof</h3>
        <p className="text-xs text-fg-faint">
          Enter a forensic bundle ID to generate a Schnorr ZK proof for chain verification.
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Bundle ID (UUID)"
            value={bundleId}
            onChange={(e) => setBundleId(e.target.value)}
            className="flex-1 ff-input font-mono"
          />
          <LoadingButton
            onClick={generateProof}
            disabled={!bundleId.trim()}
            loading={generating}
            loadingText="Proving…"
          >
            Generate Proof
          </LoadingButton>
        </div>
        {proof && <ProofBreakdown proof={proof} />}
      </div>

      {/* Verify Proof */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Verify Proof</h3>
        <p className="text-xs text-fg-faint">
          Paste or edit the proof JSON below to verify its cryptographic validity.
        </p>
        <label className="block">
          <span className="text-xs text-fg-faint mb-1 block">Proof JSON</span>
          <textarea
            placeholder='{"scheme": "schnorr", "commitment": "...", ...}'
            value={verifyInput}
            onChange={(e) => setVerifyInput(e.target.value)}
            className="w-full ff-textarea font-mono h-28 text-sm"
          />
        </label>
        <LoadingButton
          onClick={verifyProof}
          disabled={!verifyInput.trim()}
          loading={verifying}
          loadingText="Verifying…"
          className="btn-success"
        >
          Verify
        </LoadingButton>
        {verifyResult && (
          <div
            className={`rounded-lg px-4 py-3 text-sm font-medium flex items-center gap-2 ${
              verifyResult.valid
                ? "bg-teal/10 text-teal border border-teal/20"
                : "bg-rose-ember/10 text-rose-ember border border-rose-ember/20"
            }`}
          >
            <span className="text-lg">{verifyResult.valid ? "\u2713" : "\u2717"}</span>
            <span>{verifyResult.valid ? "Valid" : "Invalid"}</span>
            <span className="text-fg-faint mx-1">—</span>
            <span>Scheme: {verifyResult.scheme}</span>
          </div>
        )}
      </div>

      {/* Get Stored Proof */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Get Stored Proof</h3>
        <p className="text-xs text-fg-faint">
          Look up a previously generated proof by bundle ID.
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Bundle ID (UUID)"
            value={bundleId}
            onChange={(e) => setBundleId(e.target.value)}
            className="flex-1 ff-input font-mono"
          />
          <LoadingButton
            onClick={getStoredProof}
            disabled={!bundleId.trim()}
            loading={fetching}
            loadingText="Fetching…"
            className="btn-ghost"
          >
            Get Stored
          </LoadingButton>
        </div>
        {storedProof && <ProofBreakdown proof={storedProof as ProofData} label="Stored Proof" />}
      </div>
    </div>
  );
}
