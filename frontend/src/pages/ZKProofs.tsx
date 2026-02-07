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
      <h2 className="text-xl font-bold">ZK Proof Viewer</h2>
      {error && <div className="text-red-600 text-sm">{error}</div>}

      {/* Generate */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Generate Proof</h3>
        <div className="flex gap-2">
          <input type="text" placeholder="Bundle ID" value={bundleId} onChange={(e) => setBundleId(e.target.value)} className="flex-1 border rounded px-3 py-1 text-sm font-mono" />
          <button onClick={generateProof} className="px-4 py-1 bg-blue-600 text-white rounded text-sm">Prove</button>
          <button onClick={getStoredProof} className="px-4 py-1 bg-gray-600 text-white rounded text-sm">Get Stored</button>
        </div>
        {proof && (
          <div className="relative">
            <pre className="text-xs bg-gray-900 text-green-400 rounded p-3 overflow-x-auto">{JSON.stringify(proof, null, 2)}</pre>
            <button onClick={() => navigator.clipboard.writeText(JSON.stringify(proof, null, 2))} className="absolute top-2 right-2 text-gray-400 hover:text-white text-xs">Copy</button>
          </div>
        )}
        {storedProof && (
          <div className="relative">
            <h4 className="text-sm font-medium text-gray-600">Stored Proof</h4>
            <pre className="text-xs bg-gray-900 text-green-400 rounded p-3 overflow-x-auto">{JSON.stringify(storedProof, null, 2)}</pre>
          </div>
        )}
      </div>

      {/* Verify */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Verify Proof</h3>
        <textarea placeholder="Paste proof JSON" value={verifyInput} onChange={(e) => setVerifyInput(e.target.value)} className="w-full border rounded px-3 py-1 text-sm font-mono h-24" />
        <button onClick={verifyProof} className="px-4 py-1 bg-green-600 text-white rounded text-sm">Verify</button>
        {verifyResult && (
          <div className={`rounded p-2 text-sm ${verifyResult.valid ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"}`}>
            {verifyResult.valid ? "✓ Valid" : "✗ Invalid"} — Scheme: {verifyResult.scheme}
          </div>
        )}
      </div>
    </div>
  );
}
