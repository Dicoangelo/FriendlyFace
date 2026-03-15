import { useEffect, useState } from "react";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import ConfirmDialog from "../components/ConfirmDialog";
import { SkeletonCard, SkeletonRow } from "../components/Skeleton";

interface BackupEntry {
  filename: string;
  size_bytes: number;
  created_at: string;
  label?: string;
}

interface MigrationStatus {
  applied: string[];
  pending: string[];
  current_version: string;
}

interface SystemInfo {
  storage_backend: string;
  total_events: number;
  total_bundles: number;
  chain_integrity: { valid: boolean; count: number };
}

export default function AdminOps() {
  const [backups, setBackups] = useState<BackupEntry[]>([]);
  const [loadingBackups, setLoadingBackups] = useState(true);
  const [migrations, setMigrations] = useState<MigrationStatus | null>(null);
  const [loadingMigrations, setLoadingMigrations] = useState(true);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loadingSystem, setLoadingSystem] = useState(true);

  // Backup actions
  const [backupLabel, setBackupLabel] = useState("");
  const [creatingBackup, setCreatingBackup] = useState(false);
  const [backupSuccess, setBackupSuccess] = useState("");
  const [backupError, setBackupError] = useState("");

  // Restore confirm
  const [restoreTarget, setRestoreTarget] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(false);
  const [restoreSuccess, setRestoreSuccess] = useState("");

  // Migration actions
  const [runningMigrations, setRunningMigrations] = useState(false);
  const [migrationSuccess, setMigrationSuccess] = useState("");
  const [migrationError, setMigrationError] = useState("");
  const [rollbackConfirm, setRollbackConfirm] = useState(false);
  const [rollingBack, setRollingBack] = useState(false);

  // System actions
  const [rebuildingMerkle, setRebuildingMerkle] = useState(false);
  const [recheckingChain, setRecheckingChain] = useState(false);
  const [systemSuccess, setSystemSuccess] = useState("");
  const [systemError, setSystemError] = useState("");

  const fetchBackups = () => {
    setLoadingBackups(true);
    fetch("/api/v1/admin/backups")
      .then((r) => r.json())
      .then((data) => { setBackups(data.backups || []); setLoadingBackups(false); })
      .catch(() => setLoadingBackups(false));
  };

  const fetchMigrations = () => {
    setLoadingMigrations(true);
    fetch("/api/v1/admin/migrations/status")
      .then((r) => r.json())
      .then((data) => { setMigrations(data); setLoadingMigrations(false); })
      .catch(() => setLoadingMigrations(false));
  };

  const fetchSystem = () => {
    setLoadingSystem(true);
    fetch("/api/v1/dashboard")
      .then((r) => r.json())
      .then((data) => {
        setSystemInfo({
          storage_backend: data.storage_backend || "sqlite",
          total_events: data.total_events || 0,
          total_bundles: data.total_bundles || 0,
          chain_integrity: data.chain_integrity || { valid: true, count: 0 },
        });
        setLoadingSystem(false);
      })
      .catch(() => setLoadingSystem(false));
  };

  useEffect(() => {
    fetchBackups();
    fetchMigrations();
    fetchSystem();
  }, []);

  const createBackup = () => {
    setBackupError(""); setBackupSuccess(""); setCreatingBackup(true);
    const params = backupLabel.trim() ? `?label=${encodeURIComponent(backupLabel.trim())}` : "";
    fetch(`/api/v1/admin/backup${params}`, { method: "POST" })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => {
        setBackupSuccess(`Backup created: ${data.filename || "ok"}`);
        setCreatingBackup(false); setBackupLabel(""); fetchBackups();
      })
      .catch((e) => { setBackupError(`Backup error: ${e.message}`); setCreatingBackup(false); });
  };

  const restoreBackup = () => {
    if (!restoreTarget) return;
    setRestoring(true); setBackupError(""); setRestoreSuccess("");
    fetch(`/api/v1/admin/backup/restore?filename=${encodeURIComponent(restoreTarget)}`, { method: "POST" })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(() => {
        setRestoreSuccess("Database restored successfully");
        setRestoring(false); setRestoreTarget(null); fetchBackups(); fetchSystem();
      })
      .catch((e) => { setBackupError(`Restore error: ${e.message}`); setRestoring(false); setRestoreTarget(null); });
  };

  const runMigrations = () => {
    setMigrationError(""); setMigrationSuccess(""); setRunningMigrations(true);
    fetch("/api/v1/admin/migrations/run", { method: "POST" })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => {
        setMigrationSuccess(`Migrations applied: ${data.applied?.length || 0}`);
        setRunningMigrations(false); fetchMigrations();
      })
      .catch((e) => { setMigrationError(`Migration error: ${e.message}`); setRunningMigrations(false); });
  };

  const rollbackMigration = () => {
    setMigrationError(""); setMigrationSuccess(""); setRollingBack(true); setRollbackConfirm(false);
    fetch("/api/v1/admin/migrations/rollback", { method: "POST" })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => {
        setMigrationSuccess(`Rolled back: ${data.rolled_back || "last migration"}`);
        setRollingBack(false); fetchMigrations();
      })
      .catch((e) => { setMigrationError(`Rollback error: ${e.message}`); setRollingBack(false); });
  };

  const rebuildMerkle = () => {
    setSystemError(""); setSystemSuccess(""); setRebuildingMerkle(true);
    fetch("/api/v1/merkle/rebuild", { method: "POST" })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => {
        setSystemSuccess(`Merkle tree rebuilt: ${data.leaf_count || 0} leaves`);
        setRebuildingMerkle(false);
      })
      .catch((e) => { setSystemError(`Rebuild error: ${e.message}`); setRebuildingMerkle(false); });
  };

  const recheckChain = () => {
    setSystemError(""); setSystemSuccess(""); setRecheckingChain(true);
    fetch("/api/v1/chain/integrity")
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => {
        setSystemSuccess(`Chain integrity: ${data.valid ? "Valid" : "INVALID"} — ${data.count || 0} events`);
        setRecheckingChain(false); fetchSystem();
      })
      .catch((e) => { setSystemError(`Verify error: ${e.message}`); setRecheckingChain(false); });
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-6">
      {/* Restore confirm dialog */}
      <ConfirmDialog
        open={restoreTarget !== null}
        title="Restore Database"
        message={`This will replace the current database with backup "${restoreTarget}". This action cannot be undone.`}
        variant="danger"
        confirmLabel={restoring ? "Restoring..." : "Restore"}
        onConfirm={restoreBackup}
        onCancel={() => setRestoreTarget(null)}
      />

      {/* Rollback confirm dialog */}
      <ConfirmDialog
        open={rollbackConfirm}
        title="Rollback Migration"
        message="This will rollback the last applied migration. Make sure you have a backup."
        variant="danger"
        confirmLabel="Rollback"
        onConfirm={rollbackMigration}
        onCancel={() => setRollbackConfirm(false)}
      />

      {/* ── BACKUPS ── */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary text-lg">Backups</h3>

        {backupError && <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">{backupError}</div>}
        {backupSuccess && <div className="bg-teal/10 border border-teal/20 rounded-lg px-3 py-2 text-teal text-sm animate-fade-in">{backupSuccess}</div>}
        {restoreSuccess && <div className="bg-gold/10 border border-gold/20 rounded-lg px-3 py-2 text-gold text-sm animate-fade-in">{restoreSuccess}</div>}

        <div className="flex gap-3 items-end">
          <label className="text-sm text-fg-secondary">
            Label (optional)
            <input type="text" value={backupLabel} onChange={(e) => setBackupLabel(e.target.value)} className="ff-input w-48 block mt-1" placeholder="e.g. pre-migration" />
          </label>
          <LoadingButton onClick={createBackup} loading={creatingBackup} loadingText="Creating...">Create Backup</LoadingButton>
        </div>

        {loadingBackups ? (
          <div className="space-y-1">{[...Array(3)].map((_, i) => <SkeletonRow key={i} />)}</div>
        ) : backups.length === 0 ? (
          <EmptyState title="No backups" subtitle="Create a backup to protect your data" />
        ) : (
          <div className="space-y-2">
            {backups.map((b) => (
              <div key={b.filename} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                <div>
                  <p className="text-sm text-fg-secondary font-mono">{b.filename}</p>
                  <p className="text-xs text-fg-faint">{formatBytes(b.size_bytes)} — {new Date(b.created_at).toLocaleString()}</p>
                </div>
                <button
                  onClick={() => setRestoreTarget(b.filename)}
                  className="text-xs text-gold hover:text-gold/80 px-2 py-1 rounded hover:bg-gold/10 transition-colors"
                >
                  Restore
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── MIGRATIONS ── */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary text-lg">Migrations</h3>

        {migrationError && <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">{migrationError}</div>}
        {migrationSuccess && <div className="bg-teal/10 border border-teal/20 rounded-lg px-3 py-2 text-teal text-sm animate-fade-in">{migrationSuccess}</div>}

        {loadingMigrations ? (
          <SkeletonCard className="h-20" />
        ) : !migrations ? (
          <p className="text-fg-faint text-sm">Could not load migration status</p>
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

            <div className="flex gap-2">
              <LoadingButton
                onClick={runMigrations}
                loading={runningMigrations}
                disabled={!migrations.pending || migrations.pending.length === 0}
                loadingText="Running..."
              >
                Run Migrations
              </LoadingButton>
              <LoadingButton
                onClick={() => setRollbackConfirm(true)}
                loading={rollingBack}
                disabled={!migrations.applied || migrations.applied.length === 0}
                className="btn-ghost border border-gold/20 text-gold"
                loadingText="Rolling back..."
              >
                Rollback Last
              </LoadingButton>
            </div>
          </div>
        )}
      </div>

      {/* ── SYSTEM ── */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary text-lg">System</h3>

        {systemError && <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">{systemError}</div>}
        {systemSuccess && <div className="bg-teal/10 border border-teal/20 rounded-lg px-3 py-2 text-teal text-sm animate-fade-in">{systemSuccess}</div>}

        {loadingSystem ? (
          <SkeletonCard className="h-24" />
        ) : systemInfo ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-fg-muted">Storage Backend</p>
                <p className="text-sm font-semibold text-fg-secondary">{systemInfo.storage_backend}</p>
              </div>
              <div>
                <p className="text-xs text-fg-muted">Event Count</p>
                <p className="text-lg font-bold text-cyan">{systemInfo.total_events}</p>
              </div>
              <div>
                <p className="text-xs text-fg-muted">Bundle Count</p>
                <p className="text-lg font-bold text-amethyst">{systemInfo.total_bundles}</p>
              </div>
              <div>
                <p className="text-xs text-fg-muted">Chain Integrity</p>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className={`w-2.5 h-2.5 rounded-full ${systemInfo.chain_integrity.valid ? "bg-teal" : "bg-rose-ember"}`} />
                  <span className={`text-sm font-semibold ${systemInfo.chain_integrity.valid ? "text-teal" : "text-rose-ember"}`}>
                    {systemInfo.chain_integrity.valid ? "Valid" : "Invalid"}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <LoadingButton onClick={rebuildMerkle} loading={rebuildingMerkle} className="btn-accent" loadingText="Rebuilding...">
                Rebuild Merkle Tree
              </LoadingButton>
              <LoadingButton onClick={recheckChain} loading={recheckingChain} className="btn-ghost" loadingText="Checking...">
                Re-check Chain Integrity
              </LoadingButton>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
