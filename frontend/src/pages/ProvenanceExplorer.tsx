import { useState } from "react";

interface ProvenanceNode {
  id: string;
  entity_type: string;
  entity_id: string;
  parents: string[];
  relations: Array<{ target_id: string; relation_type: string }>;
  metadata: Record<string, unknown>;
  created_at: string;
}

export default function ProvenanceExplorer() {
  const [error, setError] = useState("");

  // Lookup chain
  const [lookupId, setLookupId] = useState("");
  const [chain, setChain] = useState<ProvenanceNode[] | null>(null);
  const [lookingUp, setLookingUp] = useState(false);

  // Add node
  const [entityType, setEntityType] = useState("model");
  const [entityId, setEntityId] = useState("");
  const [parents, setParents] = useState("");
  const [nodeMetadata, setNodeMetadata] = useState("");
  const [addResult, setAddResult] = useState<Record<string, unknown> | null>(null);
  const [adding, setAdding] = useState(false);

  const lookupChain = () => {
    if (!lookupId.trim()) {
      setError("Node ID is required");
      return;
    }
    setError("");
    setChain(null);
    setLookingUp(true);

    fetch(`/api/v1/provenance/${lookupId.trim()}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.status === 404 ? "Provenance node not found" : `${r.status}`);
        return r.json();
      })
      .then((data) => {
        setChain(Array.isArray(data) ? data : [data]);
        setLookingUp(false);
      })
      .catch((e) => {
        setError(`Lookup error: ${e.message}`);
        setLookingUp(false);
      });
  };

  const addNode = () => {
    if (!entityId.trim()) {
      setError("Entity ID is required");
      return;
    }
    setError("");
    setAddResult(null);
    setAdding(true);

    const body: Record<string, unknown> = {
      entity_type: entityType,
      entity_id: entityId.trim(),
      parents: parents.trim() ? parents.split(",").map((s) => s.trim()).filter(Boolean) : [],
      relations: [],
      metadata: {},
    };

    if (nodeMetadata.trim()) {
      try {
        body.metadata = JSON.parse(nodeMetadata);
      } catch {
        setError("Invalid JSON in metadata");
        setAdding(false);
        return;
      }
    }

    fetch("/api/v1/provenance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setAddResult(data);
        setAdding(false);
      })
      .catch((e) => {
        setError(`Add error: ${e.message}`);
        setAdding(false);
      });
  };

  const ENTITY_COLORS: Record<string, string> = {
    dataset: "bg-gold/10 text-gold border-gold/20",
    model: "bg-amethyst/10 text-amethyst border-amethyst/20",
    inference: "bg-cyan/10 text-cyan border-cyan/20",
    explanation: "bg-teal/10 text-teal border-teal/20",
    bundle: "bg-amethyst/10 text-amethyst border-amethyst/20",
    training: "bg-gold/10 text-gold border-gold/20",
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Lookup chain */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Provenance Chain Lookup</h3>
        <p className="text-xs text-fg-muted">
          Trace the full lineage of any artifact: dataset → training → model → inference → explanation → bundle.
        </p>
        <div className="flex gap-3 items-end">
          <label className="text-sm text-fg-secondary flex-1">
            Node ID
            <input
              type="text"
              value={lookupId}
              onChange={(e) => setLookupId(e.target.value)}
              className="ff-input w-full font-mono text-xs block mt-1"
              placeholder="UUID of the provenance node"
            />
          </label>
          <button onClick={lookupChain} disabled={lookingUp} className="btn-primary">
            {lookingUp ? "Tracing..." : "Trace Lineage"}
          </button>
        </div>
      </div>

      {/* Chain visualization */}
      {chain && (
        <div className="glass-card p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-fg-secondary">Provenance Chain</h3>
            <span className="text-xs text-fg-faint">{chain.length} node{chain.length !== 1 ? "s" : ""}</span>
          </div>

          <div className="space-y-3">
            {chain.map((node, i) => {
              const colorCls = ENTITY_COLORS[node.entity_type] || "bg-fg/5 text-fg-secondary border-fg-faint/20";
              return (
                <div key={node.id}>
                  {i > 0 && (
                    <div className="flex justify-center py-1">
                      <svg className="w-4 h-4 text-fg-faint" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                      </svg>
                    </div>
                  )}
                  <div className={`rounded-lg border p-3 ${colorCls}`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 rounded text-xs font-medium bg-surface">
                          {node.entity_type}
                        </span>
                        <span className="text-sm font-mono">{node.entity_id.slice(0, 16)}{node.entity_id.length > 16 ? "..." : ""}</span>
                      </div>
                      <span className="text-xs opacity-70">{new Date(node.created_at).toLocaleString()}</span>
                    </div>
                    <div className="text-xs opacity-70">
                      <span>Node: <code className="font-mono">{node.id.slice(0, 12)}...</code></span>
                      {node.parents.length > 0 && (
                        <span className="ml-3">
                          Parents: {node.parents.map((p) => p.slice(0, 8)).join(", ")}
                        </span>
                      )}
                    </div>
                    {node.relations.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {node.relations.map((r, ri) => (
                          <span key={ri} className="text-xs bg-surface rounded px-1.5 py-0.5">
                            {r.relation_type} → {r.target_id.slice(0, 8)}
                          </span>
                        ))}
                      </div>
                    )}
                    {Object.keys(node.metadata).length > 0 && (
                      <details className="mt-2 text-xs">
                        <summary className="cursor-pointer opacity-70">Metadata</summary>
                        <pre className="bg-surface rounded p-2 mt-1 overflow-x-auto">
                          {JSON.stringify(node.metadata, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Add provenance node */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Add Provenance Node</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Entity Type
            <select
              value={entityType}
              onChange={(e) => setEntityType(e.target.value)}
              className="ff-select block mt-1"
            >
              <option value="dataset">dataset</option>
              <option value="training">training</option>
              <option value="model">model</option>
              <option value="inference">inference</option>
              <option value="explanation">explanation</option>
              <option value="bundle">bundle</option>
            </select>
          </label>
          <label className="text-sm text-fg-secondary">
            Entity ID
            <input
              type="text"
              value={entityId}
              onChange={(e) => setEntityId(e.target.value)}
              className="ff-input w-48 block mt-1 font-mono text-xs"
              placeholder="UUID or identifier"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Parent Node IDs (comma-separated)
            <input
              type="text"
              value={parents}
              onChange={(e) => setParents(e.target.value)}
              className="ff-input w-64 block mt-1 font-mono text-xs"
              placeholder="Optional — uuid1, uuid2"
            />
          </label>
        </div>
        <label className="text-sm text-fg-secondary block">
          Metadata (JSON)
          <input
            type="text"
            value={nodeMetadata}
            onChange={(e) => setNodeMetadata(e.target.value)}
            className="ff-input w-full block mt-1 font-mono text-xs"
            placeholder='Optional — {"key": "value"}'
          />
        </label>
        <button onClick={addNode} disabled={adding} className="btn-success">
          {adding ? "Adding..." : "Add Node"}
        </button>
        {addResult && (
          <div className="bg-teal/10 border border-teal/20 rounded-lg px-3 py-2 text-teal text-sm">
            Node created: {String(addResult.id || "ok").slice(0, 16)}...
          </div>
        )}
      </div>
    </div>
  );
}
