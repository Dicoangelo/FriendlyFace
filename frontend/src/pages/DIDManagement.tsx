import { useEffect, useState } from "react";
import { SkeletonCard } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import { useCopyToClipboard } from "../hooks/useCopyToClipboard";

interface DIDRecord {
  did: string;
  public_key_hex: string;
  created_at: string;
}

export default function DIDManagement() {
  const copy = useCopyToClipboard();

  // DID list state
  const [dids, setDids] = useState<DIDRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Create DID state
  const [creating, setCreating] = useState(false);
  const [newDid, setNewDid] = useState<DIDRecord | null>(null);

  // Resolve state
  const [resolveInput, setResolveInput] = useState("");
  const [resolving, setResolving] = useState(false);
  const [resolvedDoc, setResolvedDoc] = useState<Record<string, unknown> | null>(null);

  // Verify VC state
  const [vcJson, setVcJson] = useState("");
  const [pubKeyHex, setPubKeyHex] = useState("");
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<{ valid: boolean; legacy?: boolean } | null>(null);

  const fetchDids = () => {
    setLoading(true);
    setError("");
    fetch("/api/v1/did/list")
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load DIDs (${r.status})`);
        return r.json();
      })
      .then((data) => {
        setDids(Array.isArray(data) ? data : data.items || []);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchDids();
  }, []);

  const createDID = () => {
    setCreating(true);
    setError("");
    setNewDid(null);
    fetch("/api/v1/did/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed: null }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`Create failed (${r.status})`);
        return r.json();
      })
      .then((d) => {
        setNewDid(d);
        setDids((prev) => [d, ...prev]);
      })
      .catch((e) => setError(e.message))
      .finally(() => setCreating(false));
  };

  const resolveDID = () => {
    if (!resolveInput.trim()) return;
    setResolving(true);
    setResolvedDoc(null);
    setError("");
    fetch(`/api/v1/did/${encodeURIComponent(resolveInput.trim())}/resolve`)
      .then((r) => {
        if (!r.ok) throw new Error(`Resolve failed (${r.status})`);
        return r.json();
      })
      .then(setResolvedDoc)
      .catch((e) => setError(e.message))
      .finally(() => setResolving(false));
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
    setVerifying(true);
    fetch("/api/v1/vc/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential: parsedVC, issuer_public_key_hex: pubKeyHex }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`Verify failed (${r.status})`);
        return r.json();
      })
      .then(setVerifyResult)
      .catch((e) => setError(e.message))
      .finally(() => setVerifying(false));
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <SkeletonCard className="h-20" />
        <SkeletonCard className="h-20" />
        <SkeletonCard className="h-20" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Error banner */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/30 text-rose-ember rounded-lg px-4 py-3 text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError("")} className="text-rose-ember/70 hover:text-rose-ember ml-4">
            &times;
          </button>
        </div>
      )}

      {/* Create DID */}
      <div className="glass-card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-fg-secondary">Create DID</h3>
            <p className="text-xs text-fg-faint">Generate an Ed25519 DID:key identity for the forensic chain.</p>
          </div>
          <LoadingButton onClick={createDID} loading={creating} loadingText="Creating...">
            Create DID
          </LoadingButton>
        </div>

        {/* Reveal card for newly created DID */}
        {newDid && (
          <div className="bg-teal/5 border border-teal/20 rounded-lg p-4 space-y-2 animate-fade-in">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-teal uppercase tracking-wider">New DID Created</span>
              <button onClick={() => setNewDid(null)} className="text-fg-faint hover:text-fg-secondary text-xs">
                Dismiss
              </button>
            </div>
            <div className="space-y-1">
              <label className="text-xs text-fg-muted">DID</label>
              <div className="flex items-center gap-2">
                <code className="text-xs font-mono text-fg-secondary bg-surface rounded px-2 py-1 break-all flex-1">
                  {newDid.did}
                </code>
                <button onClick={() => copy(newDid.did, "DID copied")} className="btn-ghost text-xs shrink-0">
                  Copy
                </button>
              </div>
            </div>
            <div className="space-y-1">
              <label className="text-xs text-fg-muted">Public Key (hex)</label>
              <div className="flex items-center gap-2">
                <code className="text-xs font-mono text-fg-secondary bg-surface rounded px-2 py-1 break-all flex-1">
                  {newDid.public_key_hex}
                </code>
                <button onClick={() => copy(newDid.public_key_hex, "Public key copied")} className="btn-ghost text-xs shrink-0">
                  Copy
                </button>
              </div>
            </div>
            {newDid.created_at && (
              <p className="text-xs text-fg-faint">Created: {new Date(newDid.created_at).toLocaleString()}</p>
            )}
          </div>
        )}
      </div>

      {/* DID list or empty state */}
      {dids.length === 0 ? (
        <div className="glass-card">
          <EmptyState
            title="No DIDs created yet"
            subtitle="Create your first decentralized identity to begin signing forensic artifacts"
            action={
              <button onClick={createDID} className="btn-primary">
                Create your first DID
              </button>
            }
          />
        </div>
      ) : (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-3">DIDs ({dids.length})</h3>
          <div className="space-y-2">
            {dids.map((d) => (
              <div key={d.did} className="flex items-center justify-between bg-surface rounded-lg p-3 text-sm">
                <div className="min-w-0 flex-1">
                  <button
                    onClick={() => copy(d.did, "DID copied")}
                    className="font-mono text-xs text-fg-secondary hover:text-cyan transition-colors cursor-pointer truncate block max-w-full text-left"
                    title="Click to copy full DID"
                  >
                    {d.did.length > 40 ? `${d.did.slice(0, 20)}...${d.did.slice(-12)}` : d.did}
                  </button>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-xs text-fg-faint">
                      Key: {d.public_key_hex.slice(0, 16)}...
                    </span>
                    {d.created_at && (
                      <span className="text-xs text-fg-faint">
                        {new Date(d.created_at).toLocaleString()}
                      </span>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => {
                    setResolveInput(d.did);
                    setResolvedDoc(null);
                    setError("");
                    setResolving(true);
                    fetch(`/api/v1/did/${encodeURIComponent(d.did)}/resolve`)
                      .then((r) => {
                        if (!r.ok) throw new Error(`Resolve failed (${r.status})`);
                        return r.json();
                      })
                      .then(setResolvedDoc)
                      .catch((e) => setError(e.message))
                      .finally(() => setResolving(false));
                  }}
                  className="btn-ghost text-xs ml-2 shrink-0"
                >
                  Resolve
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Resolve DID */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Resolve DID</h3>
        <p className="text-xs text-fg-faint">Enter a DID to fetch and display its resolution document.</p>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="did:key:z6Mk..."
            value={resolveInput}
            onChange={(e) => setResolveInput(e.target.value)}
            className="flex-1 bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm font-mono text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40"
            onKeyDown={(e) => e.key === "Enter" && resolveDID()}
          />
          <LoadingButton onClick={resolveDID} loading={resolving} loadingText="Resolving..." disabled={!resolveInput.trim()}>
            Resolve
          </LoadingButton>
        </div>
        {resolvedDoc && (
          <pre className="text-xs font-mono bg-surface rounded-lg p-3 overflow-x-auto max-h-64 text-fg-secondary">
            {JSON.stringify(resolvedDoc, null, 2)}
          </pre>
        )}
      </div>

      {/* Verify VC */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Verify Credential</h3>
        <p className="text-xs text-fg-faint">Paste a Verifiable Credential JSON to verify its signature.</p>
        <textarea
          placeholder="Paste credential JSON..."
          value={vcJson}
          onChange={(e) => setVcJson(e.target.value)}
          className="w-full bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm font-mono text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40 h-24"
        />
        <input
          type="text"
          placeholder="Issuer public key (hex)"
          value={pubKeyHex}
          onChange={(e) => setPubKeyHex(e.target.value)}
          className="w-full bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm font-mono text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40"
        />
        <LoadingButton
          onClick={verifyCredential}
          loading={verifying}
          loadingText="Verifying..."
          className="btn-accent"
          disabled={!vcJson.trim()}
        >
          Verify
        </LoadingButton>
        {verifyResult && (
          <div className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium ${verifyResult.valid ? "bg-teal/10 text-teal border border-teal/20" : "bg-rose-ember/10 text-rose-ember border border-rose-ember/20"}`}>
            <span className={`w-2 h-2 rounded-full ${verifyResult.valid ? "bg-teal" : "bg-rose-ember"}`} />
            {verifyResult.valid ? "Valid" : "Invalid"}
            {verifyResult.legacy && " (legacy format)"}
          </div>
        )}
      </div>
    </div>
  );
}
