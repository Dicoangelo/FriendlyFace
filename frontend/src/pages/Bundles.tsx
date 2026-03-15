import { useEffect, useState } from "react";
import { SkeletonCard } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import { useCopyToClipboard } from "../hooks/useCopyToClipboard";

interface BundleSummary {
  id: string;
  created_at: string;
  status: string;
  bundle_hash: string;
  merkle_root: string;
  event_count: number;
  zk_proof_placeholder?: string;
  did_credential_placeholder?: string;
}

export default function Bundles() {
  const copy = useCopyToClipboard();
  const [bundles, setBundles] = useState<BundleSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  // Create form state
  const [showCreate, setShowCreate] = useState(false);
  const [eventIdsText, setEventIdsText] = useState("");
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);

  // Expand / export state
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [exportData, setExportData] = useState<Record<string, unknown> | null>(null);
  const [exportLoading, setExportLoading] = useState(false);

  const fetchBundles = () => {
    setLoading(true);
    setError("");
    fetch("/api/v1/bundles?limit=50")
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load bundles (${r.status})`);
        return r.json();
      })
      .then((data) => {
        setBundles(data.items || []);
        setTotal(data.total || 0);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchBundles();
  }, []);

  const handleCreate = () => {
    const ids = eventIdsText
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (ids.length === 0) {
      setError("Please enter at least one event ID");
      return;
    }
    setCreating(true);
    setError("");
    setSuccessMsg("");
    fetch("/api/v1/bundles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_ids: ids }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`Create failed (${r.status})`);
        return r.json();
      })
      .then((data) => {
        setSuccessMsg(`Bundle created: ${data.id}`);
        setShowCreate(false);
        setEventIdsText("");
        setDescription("");
        fetchBundles();
      })
      .catch((e) => setError(e.message))
      .finally(() => setCreating(false));
  };

  const toggleExpand = (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      setExportData(null);
      return;
    }
    setExpandedId(id);
    setExportData(null);
    setExportLoading(true);
    fetch(`/api/v1/bundles/${id}/export`)
      .then((r) => {
        if (!r.ok) throw new Error(`Export failed (${r.status})`);
        return r.json();
      })
      .then(setExportData)
      .catch((e) => setError(e.message))
      .finally(() => setExportLoading(false));
  };

  const downloadBundle = (id: string) => {
    if (!exportData) return;
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/ld+json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bundle-${id.slice(0, 8)}.jsonld`;
    a.click();
    URL.revokeObjectURL(url);
  };

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

      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-fg-muted font-medium">{total} bundles</span>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="btn-primary"
        >
          {showCreate ? "Cancel" : "Create Bundle"}
        </button>
      </div>

      {/* Inline create form */}
      {showCreate && (
        <div className="glass-card p-4 space-y-3 animate-fade-in">
          <h3 className="text-sm font-semibold text-fg-secondary">New Bundle</h3>
          <div>
            <label className="block text-xs text-fg-muted mb-1">Event IDs (one per line or comma-separated)</label>
            <textarea
              value={eventIdsText}
              onChange={(e) => setEventIdsText(e.target.value)}
              rows={4}
              className="w-full bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm font-mono text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40"
              placeholder="e.g. 550e8400-e29b-41d4-a716-446655440000"
            />
          </div>
          <div>
            <label className="block text-xs text-fg-muted mb-1">Description (optional)</label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full bg-surface border border-border-theme rounded-lg px-3 py-2 text-sm text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-1 focus:ring-cyan/40"
              placeholder="Bundle description..."
            />
          </div>
          <LoadingButton
            onClick={handleCreate}
            loading={creating}
            loadingText="Creating..."
          >
            Create Bundle
          </LoadingButton>
        </div>
      )}

      {/* Bundle list or empty state */}
      {bundles.length === 0 ? (
        <div className="glass-card">
          <EmptyState
            title="No bundles created yet"
            subtitle="Package forensic events into cryptographically signed bundles"
            action={
              <button onClick={() => setShowCreate(true)} className="btn-primary">
                Create your first bundle
              </button>
            }
          />
        </div>
      ) : (
        <div className="space-y-2">
          {bundles.map((b) => (
            <div key={b.id} className="glass-card p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => copy(b.id, "Bundle ID copied")}
                    className="font-mono text-sm text-fg-secondary hover:text-cyan transition-colors cursor-pointer"
                    title="Click to copy full ID"
                  >
                    {b.id.slice(0, 12)}...
                  </button>
                  <StatusBadge status={b.status} />
                  {b.did_credential_placeholder && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-violet-500/10 text-violet-400">
                      DID
                    </span>
                  )}
                  {b.zk_proof_placeholder && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-500/10 text-amber-400">
                      VC
                    </span>
                  )}
                </div>
                <button
                  onClick={() => toggleExpand(b.id)}
                  className="text-xs text-fg-muted hover:text-fg-secondary transition-colors"
                >
                  {expandedId === b.id ? "Hide" : "View"}
                </button>
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className="text-xs text-fg-faint">
                  {new Date(b.created_at).toLocaleString()}
                </span>
                <span className="text-xs text-fg-muted">
                  {b.event_count} event{b.event_count !== 1 ? "s" : ""}
                </span>
              </div>

              {/* Expanded JSON-LD preview */}
              {expandedId === b.id && (
                <div className="mt-3 pt-3 border-t border-border-theme animate-fade-in">
                  {exportLoading ? (
                    <div className="text-xs text-fg-faint py-4 text-center">Loading export...</div>
                  ) : exportData ? (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-fg-muted">JSON-LD Export</span>
                        <button
                          onClick={() => downloadBundle(b.id)}
                          className="btn-accent text-xs px-2 py-1"
                        >
                          Download
                        </button>
                      </div>
                      <pre className="bg-surface rounded-lg p-3 text-xs font-mono text-fg-secondary max-h-64 overflow-auto whitespace-pre-wrap">
                        {JSON.stringify(exportData, null, 2)}
                      </pre>
                    </div>
                  ) : (
                    <div className="text-xs text-fg-faint py-2">Failed to load export data</div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
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
