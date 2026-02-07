import { useState } from "react";

export default function DIDManagement() {
  const [dids, setDids] = useState<Array<{ did: string; public_key_hex: string; created_at: string }>>([]);
  const [seed, setSeed] = useState("");
  const [resolvedDoc, setResolvedDoc] = useState<Record<string, unknown> | null>(null);

  // VC state
  const [issuerDid, setIssuerDid] = useState("");
  const [subjectDid, setSubjectDid] = useState("");
  const [claims, setClaims] = useState('{"name": "test"}');
  const [credType, setCredType] = useState("ForensicCredential");
  const [issuedVC, setIssuedVC] = useState<Record<string, unknown> | null>(null);

  // Verify VC state
  const [vcJson, setVcJson] = useState("");
  const [pubKeyHex, setPubKeyHex] = useState("");
  const [verifyResult, setVerifyResult] = useState<{ valid: boolean; legacy?: boolean } | null>(null);

  const [error, setError] = useState("");

  const createDID = () => {
    setError("");
    const body: Record<string, string | null> = { seed: seed || null };
    fetch("/api/v1/did/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((d) => setDids((prev) => [d, ...prev]))
      .catch((e) => setError(e.message));
  };

  const resolveDID = (did: string) => {
    setResolvedDoc(null);
    fetch(`/api/v1/did/${encodeURIComponent(did)}/resolve`)
      .then((r) => r.json())
      .then(setResolvedDoc)
      .catch((e) => setError(e.message));
  };

  const issueCredential = () => {
    setError("");
    setIssuedVC(null);
    let parsedClaims;
    try {
      parsedClaims = JSON.parse(claims);
    } catch {
      setError("Invalid JSON claims");
      return;
    }
    fetch("/api/v1/vc/issue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        issuer_did_id: issuerDid,
        subject_did: subjectDid,
        claims: parsedClaims,
        credential_type: credType,
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setIssuedVC)
      .catch((e) => setError(e.message));
  };

  const verifyCredential = () => {
    setError("");
    setVerifyResult(null);
    let parsedVC;
    try {
      parsedVC = JSON.parse(vcJson);
    } catch {
      setError("Invalid JSON credential");
      return;
    }
    fetch("/api/v1/vc/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential: parsedVC, issuer_public_key_hex: pubKeyHex }),
    })
      .then((r) => r.json())
      .then(setVerifyResult)
      .catch((e) => setError(e.message));
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-fg">DID / Verifiable Credentials</h2>
      {error && <div className="text-rose-ember text-sm">{error}</div>}

      {/* Create DID */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-2">Create DID</h3>
        <div className="flex gap-2">
          <input type="text" placeholder="Optional hex seed (64 chars)" value={seed} onChange={(e) => setSeed(e.target.value)} className="flex-1 ff-input font-mono" />
          <button onClick={createDID} className="btn-primary">Create</button>
        </div>
      </div>

      {/* DID list */}
      {dids.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-2">Created DIDs</h3>
          <div className="space-y-2">
            {dids.map((d) => (
              <div key={d.did} className="flex items-center justify-between bg-surface rounded-lg p-2 text-sm">
                <div>
                  <p className="font-mono text-xs">{d.did}</p>
                  <p className="text-fg-faint text-xs">Key: {d.public_key_hex.slice(0, 16)}...</p>
                </div>
                <button onClick={() => resolveDID(d.did)} className="btn-ghost">Resolve</button>
              </div>
            ))}
          </div>
        </div>
      )}

      {resolvedDoc && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-2">DID Document</h3>
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">{JSON.stringify(resolvedDoc, null, 2)}</pre>
        </div>
      )}

      {/* Issue VC */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Issue Credential</h3>
        <input type="text" placeholder="Issuer DID" value={issuerDid} onChange={(e) => setIssuerDid(e.target.value)} className="w-full ff-input font-mono" />
        <input type="text" placeholder="Subject DID (optional)" value={subjectDid} onChange={(e) => setSubjectDid(e.target.value)} className="w-full ff-input font-mono" />
        <textarea placeholder='Claims JSON: {"name": "test"}' value={claims} onChange={(e) => setClaims(e.target.value)} className="w-full ff-textarea font-mono h-20" />
        <select value={credType} onChange={(e) => setCredType(e.target.value)} className="ff-select">
          <option>ForensicCredential</option>
          <option>FLParticipantCredential</option>
          <option>AuditCredential</option>
        </select>
        <button onClick={issueCredential} className="btn-accent">Issue</button>
        {issuedVC && <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">{JSON.stringify(issuedVC, null, 2)}</pre>}
      </div>

      {/* Verify VC */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Verify Credential</h3>
        <textarea placeholder="Paste credential JSON" value={vcJson} onChange={(e) => setVcJson(e.target.value)} className="w-full ff-textarea font-mono h-20" />
        <input type="text" placeholder="Issuer public key (hex)" value={pubKeyHex} onChange={(e) => setPubKeyHex(e.target.value)} className="w-full ff-input font-mono" />
        <button onClick={verifyCredential} className="btn-success">Verify</button>
        {verifyResult && (
          <div className={`rounded p-2 text-sm ${verifyResult.valid ? "bg-teal/10 text-teal" : "bg-rose-ember/10 text-rose-ember"}`}>
            {verifyResult.valid ? "Valid" : "Invalid"}
            {verifyResult.legacy && " (legacy format)"}
          </div>
        )}
      </div>
    </div>
  );
}
