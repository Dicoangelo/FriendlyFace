import { useEffect, useState } from "react";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import { SkeletonCard, SkeletonRow } from "../components/Skeleton";

interface ProvenanceNode {
  id: string;
  node_id?: string;
  node_type?: string;
  entity_type?: string;
  entity_id?: string;
  parents: string[];
  relations?: Array<{ target_id: string; relation_type: string }>;
  metadata?: Record<string, unknown>;
  created_at: string;
  timestamp?: string;
}

interface ProvenanceStats {
  total_nodes: number;
  total_edges: number;
}

const NODE_TYPE_COLORS: Record<string, string> = {
  training: "bg-gold/10 text-gold border-gold/20",
  inference: "bg-cyan/10 text-cyan border-cyan/20",
  explanation: "bg-teal/10 text-teal border-teal/20",
  bundle: "bg-amethyst/10 text-amethyst border-amethyst/20",
  dataset: "bg-gold/10 text-gold border-gold/20",
  model: "bg-amethyst/10 text-amethyst border-amethyst/20",
};

export default function ProvenanceExplorer() {
  const [stats, setStats] = useState<ProvenanceStats | null>(null);
  const [loadingStats, setLoadingStats] = useState(true);
  const [recentNodes, setRecentNodes] = useState<ProvenanceNode[]>([]);
  const [loadingRecent, setLoadingRecent] = useState(true);
  const [error, setError] = useState("");

  // Node search
  const [searchId, setSearchId] = useState("");
  const [searchResult, setSearchResult] = useState<ProvenanceNode | null>(null);
  const [searching, setSearching] = useState(false);

  // Lineage trace
  const [lineage, setLineage] = useState<ProvenanceNode[] | null>(null);
  const [tracingLineage, setTracingLineage] = useState(false);

  useEffect(() => {
    fetch("/api/v1/provenance/stats")
      .then((r) => r.json())
      .then((data) => {
        setStats(data);
        setLoadingStats(false);
      })
      .catch(() => setLoadingStats(false));

    fetch("/api/v1/provenance/recent?limit=10")
      .then((r) => r.json())
      .then((data) => {
        setRecentNodes(data.nodes || data.items || (Array.isArray(data) ? data : []));
        setLoadingRecent(false);
      })
      .catch(() => setLoadingRecent(false));
  }, []);

  const searchNode = () => {
    if (!searchId.trim()) {
      setError("Node or event ID is required");
      return;
    }
    setError("");
    setSearchResult(null);
    setLineage(null);
    setSearching(true);

    fetch(`/api/v1/provenance/${encodeURIComponent(searchId.trim())}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.status === 404 ? "Node not found" : `${r.status}`);
        return r.json();
      })
      .then((data) => {
        setSearchResult(Array.isArray(data) ? data[0] : data);
        setSearching(false);
      })
      .catch((e) => {
        setError(`Search error: ${e.message}`);
        setSearching(false);
      });
  };

  const traceFromRoot = (nodeId: string) => {
    setTracingLineage(true);
    setLineage(null);
    setError("");

    fetch(`/api/v1/provenance/${encodeURIComponent(nodeId)}/lineage`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setLineage(Array.isArray(data) ? data : data.path || [data]);
        setTracingLineage(false);
      })
      .catch((e) => {
        setError(`Lineage trace error: ${e.message}`);
        setTracingLineage(false);
      });
  };

  const getNodeType = (node: ProvenanceNode) => node.node_type || node.entity_type || "unknown";
  const getNodeId = (node: ProvenanceNode) => node.node_id || node.id;
  const getTimestamp = (node: ProvenanceNode) => node.timestamp || node.created_at;

  return (
    <div className="space-y-6">
      {/* Stats header */}
      {loadingStats ? (
        <div className="grid grid-cols-2 gap-4">
          <SkeletonCard className="h-20" />
          <SkeletonCard className="h-20" />
        </div>
      ) : stats && stats.total_nodes > 0 ? (
        <div className="grid grid-cols-2 gap-4">
          <div className="glass-card p-4 border-l-2 border-cyan/20">
            <p className="text-sm text-fg-muted">Total Nodes</p>
            <p className="text-2xl font-bold text-cyan mt-1">{stats.total_nodes}</p>
          </div>
          <div className="glass-card p-4 border-l-2 border-amethyst/20">
            <p className="text-sm text-fg-muted">Total Edges</p>
            <p className="text-2xl font-bold text-amethyst mt-1">{stats.total_edges}</p>
          </div>
        </div>
      ) : !loadingStats && (!stats || stats.total_nodes === 0) ? (
        <EmptyState title="No provenance nodes" subtitle="Provenance nodes are created during training, inference, and bundling operations" />
      ) : null}

      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Node search */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Node Lookup</h3>
        <p className="text-xs text-fg-muted">
          Search by event ID or node ID to inspect provenance details.
        </p>
        <div className="flex gap-3 items-end">
          <label className="text-sm text-fg-secondary flex-1">
            Event ID or Node ID
            <input
              type="text"
              value={searchId}
              onChange={(e) => setSearchId(e.target.value)}
              className="ff-input w-full font-mono text-xs block mt-1"
              placeholder="UUID of the provenance node or event"
              onKeyDown={(e) => e.key === "Enter" && searchNode()}
            />
          </label>
          <LoadingButton onClick={searchNode} loading={searching} loadingText="Searching...">
            Search
          </LoadingButton>
        </div>
      </div>

      {/* Search result node card */}
      {searchResult && (
        <NodeCard
          node={searchResult}
          getNodeType={getNodeType}
          getNodeId={getNodeId}
          getTimestamp={getTimestamp}
          onSearchParent={(id) => { setSearchId(id); }}
          onTraceRoot={() => traceFromRoot(getNodeId(searchResult))}
          tracingLineage={tracingLineage}
        />
      )}

      {/* Lineage path */}
      {lineage && lineage.length > 0 && (
        <div className="glass-card p-4 space-y-3">
          <h3 className="font-semibold text-fg-secondary">Lineage Path</h3>
          <div className="space-y-1">
            {lineage.map((node, i) => {
              const type = getNodeType(node);
              const colorCls = NODE_TYPE_COLORS[type] || "bg-fg/5 text-fg-secondary border-fg-faint/20";
              return (
                <div key={getNodeId(node)}>
                  {i > 0 && (
                    <div className="flex justify-center py-0.5">
                      <svg className="w-4 h-4 text-fg-faint" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                      </svg>
                    </div>
                  )}
                  <div className={`rounded-lg border p-2 flex items-center justify-between ${colorCls}`}>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-surface">{type}</span>
                      <code className="text-xs font-mono">{getNodeId(node).slice(0, 16)}...</code>
                    </div>
                    <span className="text-xs opacity-70">{new Date(getTimestamp(node)).toLocaleString()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Recent nodes */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Recent Nodes</h3>
        {loadingRecent ? (
          <div className="space-y-1">
            {[...Array(5)].map((_, i) => <SkeletonRow key={i} />)}
          </div>
        ) : recentNodes.length === 0 ? (
          <EmptyState title="No recent nodes" subtitle="Nodes appear as provenance records are created" />
        ) : (
          <div className="space-y-2">
            {recentNodes.slice(0, 10).map((node) => {
              const type = getNodeType(node);
              const colorCls = NODE_TYPE_COLORS[type] || "bg-fg/5 text-fg-secondary";
              return (
                <div
                  key={getNodeId(node)}
                  className="flex items-center justify-between bg-surface rounded-lg px-3 py-2 cursor-pointer hover:bg-fg/[0.02]"
                  onClick={() => { setSearchId(getNodeId(node)); setSearchResult(node); }}
                >
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colorCls}`}>{type}</span>
                    <code className="text-xs font-mono text-fg-secondary">{getNodeId(node).slice(0, 16)}</code>
                  </div>
                  <span className="text-xs text-fg-faint">{new Date(getTimestamp(node)).toLocaleString()}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function NodeCard({
  node,
  getNodeType,
  getNodeId,
  getTimestamp,
  onSearchParent,
  onTraceRoot,
  tracingLineage,
}: {
  node: ProvenanceNode;
  getNodeType: (n: ProvenanceNode) => string;
  getNodeId: (n: ProvenanceNode) => string;
  getTimestamp: (n: ProvenanceNode) => string;
  onSearchParent: (id: string) => void;
  onTraceRoot: () => void;
  tracingLineage: boolean;
}) {
  const type = getNodeType(node);
  const colorCls = NODE_TYPE_COLORS[type] || "bg-fg/5 text-fg-secondary border-fg-faint/20";

  return (
    <div className={`glass-card p-4 border-l-2 animate-fade-in ${colorCls.split(" ").find((c) => c.startsWith("border")) || "border-fg-faint/20"}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${colorCls}`}>{type}</span>
          <h3 className="font-semibold text-fg-secondary text-sm">Node Detail</h3>
        </div>
        <LoadingButton
          onClick={onTraceRoot}
          loading={tracingLineage}
          className="btn-ghost text-xs"
          loadingText="Tracing..."
        >
          Trace from root
        </LoadingButton>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-xs text-fg-muted">Node ID</p>
          <p className="font-mono text-fg-secondary text-xs">{getNodeId(node)}</p>
        </div>
        <div>
          <p className="text-xs text-fg-muted">Timestamp</p>
          <p className="text-fg-secondary text-xs">{new Date(getTimestamp(node)).toLocaleString()}</p>
        </div>
      </div>

      {node.parents && node.parents.length > 0 && (
        <div className="mt-3">
          <p className="text-xs text-fg-muted mb-1">Parents</p>
          <div className="flex flex-wrap gap-1">
            {node.parents.map((p) => (
              <button
                key={p}
                onClick={() => onSearchParent(p)}
                className="text-xs font-mono text-cyan hover:text-cyan/80 bg-cyan/5 hover:bg-cyan/10 px-2 py-0.5 rounded transition-colors"
              >
                {p.slice(0, 12)}...
              </button>
            ))}
          </div>
        </div>
      )}

      {node.metadata && Object.keys(node.metadata).length > 0 && (
        <details className="mt-3 text-xs">
          <summary className="cursor-pointer text-fg-muted">Metadata</summary>
          <pre className="bg-surface rounded p-2 mt-1 overflow-x-auto">
            {JSON.stringify(node.metadata, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}
