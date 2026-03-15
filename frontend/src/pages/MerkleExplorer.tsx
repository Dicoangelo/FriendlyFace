import { useEffect, useState, useCallback } from "react";
import { SkeletonCard } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import { useCopyToClipboard } from "../hooks/useCopyToClipboard";

interface MerkleProof {
  leaf_hash: string;
  leaf_index: number;
  proof_hashes: string[];
  proof_directions: string[];
  root_hash: string;
}

function truncateHash(hash: string, len = 12): string {
  if (hash.length <= len * 2 + 3) return hash;
  return `${hash.slice(0, len)}...${hash.slice(-len)}`;
}

export default function MerkleExplorer() {
  const [root, setRoot] = useState<string | null>(null);
  const [leafCount, setLeafCount] = useState<number>(0);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [rootLoading, setRootLoading] = useState(true);
  const [error, setError] = useState("");
  const copy = useCopyToClipboard();

  // Proof lookup
  const [proofEventId, setProofEventId] = useState("");
  const [proof, setProof] = useState<MerkleProof | null>(null);
  const [proofLoading, setProofLoading] = useState(false);

  const fetchRoot = useCallback(() => {
    setRootLoading(true);
    setError("");
    fetch("/api/v1/merkle/root")
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch root (${r.status})`);
        return r.json();
      })
      .then((data) => {
        setRoot(data.merkle_root);
        setLeafCount(data.leaf_count ?? 0);
        setLastUpdated(new Date().toLocaleString());
        setRootLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setRootLoading(false);
      });
  }, []);

  useEffect(() => {
    fetchRoot();
  }, [fetchRoot]);

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
        if (!r.ok)
          throw new Error(
            r.status === 404
              ? "Event not found in Merkle tree"
              : `Proof lookup failed (${r.status})`,
          );
        return r.json();
      })
      .then((data) => {
        setProof(data);
        setProofLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setProofLoading(false);
      });
  };

  return (
    <div className="space-y-6">
      {/* Error banner */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm flex items-center justify-between">
          <span>{error}</span>
          <button
            onClick={() => setError("")}
            className="ml-3 text-rose-ember/70 hover:text-rose-ember"
          >
            &times;
          </button>
        </div>
      )}

      {/* Merkle root card */}
      {rootLoading ? (
        <SkeletonCard className="h-32" />
      ) : leafCount === 0 ? (
        <EmptyState
          title="No events in the Merkle tree yet"
          subtitle="Record forensic events to build the Merkle tree. Each event becomes a leaf in the hash chain."
        />
      ) : (
        <div className="glass-card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-fg-secondary">
              Merkle Tree Root
            </h3>
            <LoadingButton
              onClick={fetchRoot}
              loading={rootLoading}
              loadingText="Refreshing..."
              className="btn-secondary text-xs"
            >
              Refresh Root
            </LoadingButton>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-fg-muted">Root Hash</p>
              <div className="flex items-center gap-2 mt-1">
                <p className="text-sm font-mono text-teal">
                  {root ? truncateHash(root) : "—"}
                </p>
                {root && (
                  <button
                    onClick={() => copy(root, "Root hash copied")}
                    className="text-fg-faint hover:text-fg-secondary text-xs"
                    title="Copy full hash"
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                  </button>
                )}
              </div>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Leaf Count</p>
              <p className="text-2xl font-bold text-cyan mt-1">{leafCount}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Last Updated</p>
              <p className="text-sm text-fg-secondary mt-1">
                {lastUpdated ?? "—"}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Proof lookup */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">
          Inclusion Proof Lookup
        </h3>
        <p className="text-xs text-fg-muted">
          Verify that a specific event is included in the Merkle tree. Provides
          the cryptographic path from leaf to root.
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
          <LoadingButton
            onClick={lookupProof}
            loading={proofLoading}
            loadingText="Looking up..."
          >
            Get Proof
          </LoadingButton>
        </div>
      </div>

      {/* Proof result */}
      {proof && (
        <div className="glass-card p-4 space-y-4">
          <h3 className="font-semibold text-fg-secondary">
            Merkle Inclusion Proof
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-fg-muted">Leaf Position</p>
              <p className="text-lg font-bold text-cyan">
                Leaf {proof.leaf_index + 1} of {leafCount}
              </p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Path Length</p>
              <p className="text-lg font-bold text-gold">
                {proof.proof_hashes.length}
              </p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Root Match</p>
              <p
                className={`text-lg font-bold ${proof.root_hash === root ? "text-teal" : "text-rose-ember"}`}
              >
                {proof.root_hash === root ? "Verified" : "Mismatch"}
              </p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Leaf Hash</p>
              <div className="flex items-center gap-1 mt-0.5">
                <p className="text-xs font-mono text-cyan truncate">
                  {truncateHash(proof.leaf_hash, 8)}
                </p>
                <button
                  onClick={() => copy(proof.leaf_hash, "Leaf hash copied")}
                  className="text-fg-faint hover:text-fg-secondary flex-shrink-0"
                  title="Copy leaf hash"
                >
                  <svg
                    className="w-3 h-3"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>

          {/* Proof path — vertical chain */}
          {proof.proof_hashes.length > 0 ? (
            <div>
              <p className="text-xs font-semibold text-fg-muted mb-3">
                Proof Path (leaf &rarr; root)
              </p>
              <div className="relative pl-6">
                {proof.proof_hashes.map((hash, i) => (
                  <div key={i} className="relative pb-4 last:pb-0">
                    {/* Vertical line */}
                    {i < proof.proof_hashes.length - 1 && (
                      <div className="absolute left-[-16px] top-5 bottom-0 w-px bg-border-theme" />
                    )}
                    {/* Node dot */}
                    <div className="absolute left-[-20px] top-1.5 w-2 h-2 rounded-full bg-amethyst ring-2 ring-surface" />
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-medium ${
                          proof.proof_directions[i] === "left"
                            ? "bg-amethyst/10 text-amethyst"
                            : "bg-cyan/10 text-cyan"
                        }`}
                      >
                        {proof.proof_directions[i]}
                      </span>
                      <span className="text-xs font-mono text-fg-secondary">
                        {truncateHash(hash, 16)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-fg-muted italic">
              No sibling hashes — this leaf is the only node in the tree.
            </p>
          )}

          {/* Root hash */}
          <div className="bg-surface rounded-lg p-3">
            <p className="text-xs text-fg-muted mb-1">Root Hash</p>
            <p
              className={`text-xs font-mono break-all ${proof.root_hash === root ? "text-teal" : "text-rose-ember"}`}
            >
              {proof.root_hash}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
