import { useEffect, useState } from "react";

interface MerkleProof {
  event_id: string;
  leaf_hash: string;
  root: string;
  path: Array<{ hash: string; direction: string }>;
  leaf_index: number;
  tree_size: number;
}

export default function MerkleExplorer() {
  const [root, setRoot] = useState<string | null>(null);
  const [leafCount, setLeafCount] = useState<number>(0);
  const [error, setError] = useState("");

  // Proof lookup
  const [proofEventId, setProofEventId] = useState("");
  const [proof, setProof] = useState<MerkleProof | null>(null);
  const [proofLoading, setProofLoading] = useState(false);

  useEffect(() => {
    fetch("/api/v1/merkle/root")
      .then((r) => r.json())
      .then((data) => {
        setRoot(data.merkle_root);
        setLeafCount(data.leaf_count ?? 0);
      })
      .catch((e) => setError(e.message));
  }, []);

  const lookupProof = () => {
    if (!proofEventId.trim()) {
      setError("Event ID is required");
      return;
    }
    setError("");
    setProof(null);
    setProofLoading(true);

    fetch(`/api/v1/merkle/proof/${proofEventId.trim()}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.status === 404 ? "Event not found in Merkle tree" : `${r.status}`);
        return r.json();
      })
      .then((data) => {
        setProof(data);
        setProofLoading(false);
      })
      .catch((e) => {
        setError(`Proof error: ${e.message}`);
        setProofLoading(false);
      });
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Merkle root status */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Merkle Tree Root</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-fg-muted">Root Hash</p>
            <p className="text-sm font-mono text-teal break-all mt-1">{root || "Loading..."}</p>
          </div>
          <div>
            <p className="text-xs text-fg-muted">Leaf Count</p>
            <p className="text-2xl font-bold text-cyan">{leafCount}</p>
          </div>
        </div>
      </div>

      {/* Proof lookup */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Inclusion Proof Lookup</h3>
        <p className="text-xs text-fg-muted">
          Verify that a specific event is included in the Merkle tree. Provides the cryptographic path from leaf to root.
        </p>
        <div className="flex gap-3 items-end">
          <label className="text-sm text-fg-secondary flex-1">
            Event ID
            <input
              type="text"
              value={proofEventId}
              onChange={(e) => setProofEventId(e.target.value)}
              className="ff-input w-full font-mono text-xs block mt-1"
              placeholder="UUID of the forensic event"
            />
          </label>
          <button onClick={lookupProof} disabled={proofLoading} className="btn-primary">
            {proofLoading ? "Looking up..." : "Get Proof"}
          </button>
        </div>
      </div>

      {/* Proof visualization */}
      {proof && (
        <div className="glass-card p-4 space-y-4">
          <h3 className="font-semibold text-fg-secondary">Merkle Inclusion Proof</h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-fg-muted">Leaf Index</p>
              <p className="text-lg font-bold text-cyan">{proof.leaf_index}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Tree Size</p>
              <p className="text-lg font-bold text-amethyst">{proof.tree_size}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Path Length</p>
              <p className="text-lg font-bold text-gold">{proof.path.length}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Verified</p>
              <p className="text-lg font-bold text-teal">{proof.root === root ? "Yes" : "Mismatch!"}</p>
            </div>
          </div>

          {/* Leaf hash */}
          <div className="bg-surface rounded-lg p-3">
            <p className="text-xs text-fg-muted mb-1">Leaf Hash (Event)</p>
            <p className="text-xs font-mono text-cyan break-all">{proof.leaf_hash}</p>
          </div>

          {/* Path visualization */}
          <div className="space-y-2">
            <p className="text-xs font-semibold text-fg-muted">Proof Path (leaf → root)</p>
            {proof.path.map((step, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="flex items-center gap-2 w-16">
                  <span className="text-xs text-fg-faint">Step {i + 1}</span>
                </div>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${step.direction === "left" ? "bg-amethyst/10 text-amethyst" : "bg-cyan/10 text-cyan"}`}>
                  {step.direction}
                </span>
                <span className="text-xs font-mono text-fg-secondary break-all flex-1">{step.hash}</span>
              </div>
            ))}
          </div>

          {/* Root hash */}
          <div className="bg-surface rounded-lg p-3">
            <p className="text-xs text-fg-muted mb-1">Root Hash</p>
            <p className={`text-xs font-mono break-all ${proof.root === root ? "text-teal" : "text-rose-ember"}`}>
              {proof.root}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
