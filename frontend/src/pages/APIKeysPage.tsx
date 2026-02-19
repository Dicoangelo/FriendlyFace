import { useEffect, useState } from "react";

interface APIKey {
  id: string;
  name: string | null;
  rate_limit: number;
  created_at: string;
  revoked_at: string | null;
}

function getToken(): string | null {
  return localStorage.getItem("ff_token");
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

export default function APIKeysPage() {
  const [keys, setKeys] = useState<APIKey[]>([]);
  const [newKeyName, setNewKeyName] = useState("default");
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const fetchKeys = async () => {
    try {
      const res = await fetch("/api/v1/keys", { headers: authHeaders() });
      if (!res.ok) throw new Error("Failed to load keys");
      const data = await res.json();
      setKeys(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchKeys();
  }, []);

  const handleCreate = async () => {
    setError("");
    setCreatedKey(null);
    try {
      const res = await fetch("/api/v1/keys", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ name: newKeyName }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to create key");
      }
      const data = await res.json();
      setCreatedKey(data.key);
      setNewKeyName("default");
      fetchKeys();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleRevoke = async (keyId: string) => {
    try {
      const res = await fetch(`/api/v1/keys/${keyId}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok && res.status !== 204) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to revoke key");
      }
      fetchKeys();
    } catch (err: any) {
      setError(err.message);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-fg-muted">Loading API keys...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-fg">API Keys</h2>
          <p className="text-sm text-fg-muted mt-1">
            Manage API keys for programmatic access to FriendlyFace endpoints.
          </p>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-rose-ember/10 border border-rose-ember/20 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Create new key */}
      <div className="p-6 rounded-xl border border-border-theme bg-sidebar/50">
        <h3 className="text-sm font-medium text-fg mb-4">Create New API Key</h3>
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <label htmlFor="key-name" className="block text-xs text-fg-muted mb-1">
              Key Name
            </label>
            <input
              id="key-name"
              type="text"
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-border-theme bg-page text-fg text-sm focus:outline-none focus:border-cyan"
              placeholder="e.g., production, staging"
            />
          </div>
          <button
            onClick={handleCreate}
            className="px-4 py-2 rounded-lg bg-cyan text-white text-sm font-medium hover:bg-cyan/90 transition-colors"
          >
            Generate Key
          </button>
        </div>

        {createdKey && (
          <div className="mt-4 p-4 rounded-lg bg-teal/10 border border-teal/20">
            <p className="text-sm font-medium text-teal mb-2">
              Key created! Copy it now -- it won't be shown again.
            </p>
            <code className="block p-3 rounded bg-page text-sm text-fg font-mono break-all select-all">
              {createdKey}
            </code>
          </div>
        )}
      </div>

      {/* Key list */}
      <div className="rounded-xl border border-border-theme overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="bg-sidebar/50 border-b border-border-theme">
              <th className="text-left px-4 py-3 text-xs font-medium text-fg-muted uppercase">Name</th>
              <th className="text-left px-4 py-3 text-xs font-medium text-fg-muted uppercase">ID</th>
              <th className="text-left px-4 py-3 text-xs font-medium text-fg-muted uppercase">Rate Limit</th>
              <th className="text-left px-4 py-3 text-xs font-medium text-fg-muted uppercase">Created</th>
              <th className="text-left px-4 py-3 text-xs font-medium text-fg-muted uppercase">Status</th>
              <th className="text-right px-4 py-3 text-xs font-medium text-fg-muted uppercase">Actions</th>
            </tr>
          </thead>
          <tbody>
            {keys.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-fg-muted text-sm">
                  No API keys yet. Create one above.
                </td>
              </tr>
            ) : (
              keys.map((key) => (
                <tr key={key.id} className="border-b border-border-theme last:border-b-0 hover:bg-fg/[0.02]">
                  <td className="px-4 py-3 text-sm text-fg">{key.name || "--"}</td>
                  <td className="px-4 py-3 text-sm text-fg-muted font-mono">{key.id}</td>
                  <td className="px-4 py-3 text-sm text-fg-muted">{key.rate_limit}/min</td>
                  <td className="px-4 py-3 text-sm text-fg-muted">{key.created_at}</td>
                  <td className="px-4 py-3">
                    {key.revoked_at ? (
                      <span className="px-2 py-1 rounded text-xs bg-rose-ember/10 text-rose-ember border border-rose-ember/20">
                        Revoked
                      </span>
                    ) : (
                      <span className="px-2 py-1 rounded text-xs bg-teal/10 text-teal border border-teal/20">
                        Active
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    {!key.revoked_at && (
                      <button
                        onClick={() => handleRevoke(key.id)}
                        className="text-xs text-rose-ember hover:text-rose-ember/80 transition-colors"
                      >
                        Revoke
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
