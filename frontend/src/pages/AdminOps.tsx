import { useEffect, useState } from "react";
import EmptyState from "../components/EmptyState";

interface BackupEntry {
  filename: string;
  size_bytes: number;
  created_at: string;
  label?: string;
}

interface BackupStats {
  total_backups: number;
  total_size_bytes: number;
  oldest: string | null;
  newest: string | null;
}

interface MigrationStatus {
  applied: string[];
  pending: string[];
  current_version: string;
}

export default function AdminOps() {
  const [backups, setBackups] = useState<BackupEntry[]>([]);
  const [stats, setStats] = useState<BackupStats | null>(null);
  const [migrations, setMigrations] = useState<MigrationStatus | null>(null);
  const [error, setError] = useState("");

  // Create backup
  const [backupLabel, setBackupLabel] = useState("");
  const [creatingBackup, setCreatingBackup] = useState(false);
  const [backupResult, setBackupResult] = useState<Record<string, unknown> | null>(null);

  // Verify & Restore
  const [verifyResult, setVerifyResult] = useState<Record<string, unknown> | null>(null);
  const [restoring, setRestoring] = useState<string | null>(null);
  const [restoreResult, setRestoreResult] = useState<Record<string, unknown> | null>(null);

  // Rollback
  const [rollingBack, setRollingBack] = useState(false);
  const [rollbackResult, setRollbackResult] = useState<Record<string, unknown> | null>(null);

  const fetchAll = () => {
    fetch("/api/v1/admin/backups")
      .then((r) => r.json())
      .then((data) => setBackups(data.backups || []))
      .catch(() => {});

    fetch("/api/v1/admin/backup/stats")
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {});

    fetch("/api/v1/admin/migrations/status")
      .then((r) => r.json())
      .then(setMigrations)
      .catch(() => {});
  };

  useEffect(() => {
    fetchAll();
  }, []);

  const createBackup = () => {
    setError("");
    setBackupResult(null);
    setCreatingBackup(true);

    const params = backupLabel.trim() ? `?label=${encodeURIComponent(backupLabel.trim())}` : "";
    fetch(`/api/v1/admin/backup${params}`, { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setBackupResult(data);
        setCreatingBackup(false);
        setBackupLabel("");
        fetchAll();
      })
      .catch((e) => {
        setError(`Backup error: ${e.message}`);
        setCreatingBackup(false);
      });
  };

  const verifyBackup = (filename: string) => {
    setVerifyResult(null);
    fetch(`/api/v1/admin/backup/verify?filename=${encodeURIComponent(filename)}`, { method: "POST" })
      .then((r) => r.json())
      .then(setVerifyResult)
      .catch((e) => setError(`Verify error: ${e.message}`));
  };

  const restoreBackup = (filename: string) => {
    setRestoring(filename);
    setRestoreResult(null);
    fetch(`/api/v1/admin/backup/restore?filename=${encodeURIComponent(filename)}`, { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setRestoreResult(data);
        setRestoring(null);
        fetchAll();
      })
      .catch((e) => {
        setError(`Restore error: ${e.message}`);
        setRestoring(null);
      });
  };

  const rollbackMigration = () => {
    setRollingBack(true);
    setRollbackResult(null);
    fetch("/api/v1/admin/migrations/rollback", { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setRollbackResult(data);
        setRollingBack(false);
        fetchAll();
      })
      .catch((e) => {
        setError(`Rollback error: ${e.message}`);
        setRollingBack(false);
      });
  };

  const cleanupBackups = () => {
    fetch("/api/v1/admin/backup/cleanup", { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        fetchAll();
      })
      .catch((e) => setError(`Cleanup error: ${e.message}`));
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Stats banner */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Total Backups" value={stats.total_backups} color="cyan" />
          <StatCard label="Total Size" value={formatBytes(stats.total_size_bytes)} color="amethyst" />
          <StatCard label="Oldest" value={stats.oldest ? new Date(stats.oldest).toLocaleDateString() : "—"} color="gold" />
          <StatCard label="Newest" value={stats.newest ? new Date(stats.newest).toLocaleDateString() : "—"} color="teal" />
        </div>
      )}

      {/* Create backup */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Create Backup</h3>
        <div className="flex gap-3 items-end">
          <label className="text-sm text-fg-secondary">
            Label (optional)
            <input
              type="text"
              value={backupLabel}
              onChange={(e) => setBackupLabel(e.target.value)}
              className="ff-input w-48 block mt-1"
              placeholder="e.g. pre-migration"
            />
          </label>
          <button onClick={createBackup} disabled={creatingBackup} className="btn-primary">
            {creatingBackup ? "Creating..." : "Create Backup"}
          </button>
          <button onClick={cleanupBackups} className="btn-ghost text-sm">
            Cleanup Old
          </button>
        </div>
        {backupResult && (
          <div className="bg-teal/10 border border-teal/20 rounded-lg px-3 py-2 text-teal text-sm animate-fade-in">
            Backup created: {String(backupResult.filename || "ok")}
          </div>
        )}
      </div>

      {/* Verify result */}
      {verifyResult && (
        <div className="glass-card p-4 animate-fade-in">
          <h3 className="font-semibold text-fg-secondary mb-2">Verification Result</h3>
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
            {JSON.stringify(verifyResult, null, 2)}
          </pre>
        </div>
      )}

      {/* Restore result */}
      {restoreResult && (
        <div className="glass-card p-4 animate-fade-in">
          <h3 className="font-semibold text-fg-secondary mb-2">Restore Result</h3>
          <div className="bg-gold/10 border border-gold/20 rounded-lg px-3 py-2 text-gold text-sm">
            Database restored successfully
          </div>
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto mt-2">
            {JSON.stringify(restoreResult, null, 2)}
          </pre>
        </div>
      )}

      {/* Backups list */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Backups</h3>
        {backups.length === 0 ? (
          <EmptyState title="No backups" subtitle="Create a backup to protect your data" />
        ) : (
          <div className="space-y-2">
            {backups.map((b) => (
              <div key={b.filename} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                <div>
                  <p className="text-sm text-fg-secondary font-mono">{b.filename}</p>
                  <p className="text-xs text-fg-faint">
                    {formatBytes(b.size_bytes)}
                    {b.label && <> — <span className="text-fg-muted">{b.label}</span></>}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => verifyBackup(b.filename)}
                    className="text-xs text-cyan hover:text-cyan/80 px-2 py-1 rounded hover:bg-cyan/10 transition-colors"
                  >
                    Verify
                  </button>
                  <button
                    onClick={() => restoreBackup(b.filename)}
                    disabled={restoring === b.filename}
                    className="text-xs text-gold hover:text-gold/80 px-2 py-1 rounded hover:bg-gold/10 transition-colors"
                  >
                    {restoring === b.filename ? "Restoring..." : "Restore"}
                  </button>
                  <span className="text-xs text-fg-faint">{new Date(b.created_at).toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Migrations */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-fg-secondary">Database Migrations</h3>
          <button
            onClick={rollbackMigration}
            disabled={rollingBack || !migrations || migrations.applied.length === 0}
            className="text-xs text-gold hover:text-gold/80 px-3 py-1 rounded border border-gold/20 hover:bg-gold/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {rollingBack ? "Rolling back..." : "Rollback Last"}
          </button>
        </div>
        {!migrations ? (
          <p className="text-fg-faint text-sm">Loading...</p>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-sm text-fg-muted">Current version:</span>
              <span className="text-sm font-mono text-fg-secondary">{migrations.current_version || "none"}</span>
            </div>
            {migrations.applied.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-teal mb-1">Applied ({migrations.applied.length})</p>
                <div className="flex flex-wrap gap-1">
                  {migrations.applied.map((m) => (
                    <span key={m} className="px-2 py-0.5 rounded text-xs font-mono bg-teal/10 text-teal">{m}</span>
                  ))}
                </div>
              </div>
            )}
            {migrations.pending.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-gold mb-1">Pending ({migrations.pending.length})</p>
                <div className="flex flex-wrap gap-1">
                  {migrations.pending.map((m) => (
                    <span key={m} className="px-2 py-0.5 rounded text-xs font-mono bg-gold/10 text-gold">{m}</span>
                  ))}
                </div>
              </div>
            )}
            {migrations.applied.length === 0 && migrations.pending.length === 0 && (
              <p className="text-fg-faint text-sm">No migrations found</p>
            )}
          </div>
        )}
      </div>
      {/* Rollback result */}
      {rollbackResult && (
        <div className="glass-card p-4 animate-fade-in">
          <h3 className="font-semibold text-fg-secondary mb-2">Rollback Result</h3>
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
            {JSON.stringify(rollbackResult, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

const STAT_COLORS: Record<string, string> = {
  cyan: "border-cyan/20 text-cyan",
  amethyst: "border-amethyst/20 text-amethyst",
  teal: "border-teal/20 text-teal",
  gold: "border-gold/20 text-gold",
};

function StatCard({ label, value, color = "cyan" }: { label: string; value: string | number; color?: string }) {
  return (
    <div className={`glass-card p-4 border-l-2 ${STAT_COLORS[color] || STAT_COLORS.cyan}`}>
      <p className="text-sm text-fg-muted">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${STAT_COLORS[color]?.split(" ")[1] || "text-cyan"}`}>{value}</p>
    </div>
  );
}
