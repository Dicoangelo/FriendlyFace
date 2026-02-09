import { useState } from "react";
import LoadingButton from "../components/LoadingButton";

interface SimResult {
  simulation_id: string;
  n_rounds: number;
  n_clients: number;
  rounds: Array<{
    round: number;
    global_model_hash: string;
    event_id: string;
    poisoning?: { flagged_client_ids: string[]; n_flagged: number };
    noise_scale?: number;
    n_clipped?: number;
    privacy_spent?: number;
  }>;
  dp_config?: { epsilon: number; delta: number; max_grad_norm: number };
  total_epsilon?: number;
}

export default function FLSimulations() {
  const [nClients, setNClients] = useState(5);
  const [nRounds, setNRounds] = useState(3);
  const [enablePoisoning, setEnablePoisoning] = useState(true);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState<SimResult | null>(null);
  const [loading, setLoading] = useState(false);

  // DP state
  const [epsilon, setEpsilon] = useState(1.0);
  const [delta, setDelta] = useState(1e-5);
  const [maxGradNorm, setMaxGradNorm] = useState(1.0);
  const [dpResult, setDpResult] = useState<SimResult | null>(null);

  // Status
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<"standard" | "dp">("standard");

  // Round inspector
  const [roundDetail, setRoundDetail] = useState<Record<string, unknown> | null>(null);
  const [roundSecurity, setRoundSecurity] = useState<Record<string, unknown> | null>(null);

  const startSimulation = () => {
    setLoading(true);
    setError("");
    fetch("/api/v1/fl/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_clients: nClients, n_rounds: nRounds, enable_poisoning_detection: enablePoisoning, seed }),
    })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => { setResult(data); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  };

  const startDPSimulation = () => {
    setLoading(true);
    setError("");
    fetch("/api/v1/fl/dp-start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_clients: nClients, n_rounds: nRounds, epsilon, delta, max_grad_norm: maxGradNorm, seed }),
    })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => { setDpResult(data); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  };

  const fetchStatus = () => {
    fetch("/api/v1/fl/status").then((r) => r.json()).then(setStatus);
  };

  const inspectRound = (simId: string, roundNum: number) => {
    setRoundDetail(null);
    setRoundSecurity(null);
    fetch(`/api/v1/fl/rounds/${simId}/${roundNum}`)
      .then((r) => r.json())
      .then(setRoundDetail)
      .catch((e) => setError(`Round detail error: ${e.message}`));
    fetch(`/api/v1/fl/rounds/${simId}/${roundNum}/security`)
      .then((r) => r.json())
      .then(setRoundSecurity)
      .catch(() => {});
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        {/* Page title shown in header bar */}
        <button onClick={fetchStatus} className="btn-ghost">Refresh Status</button>
      </div>
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2">
        <button onClick={() => setTab("standard")} className={`px-4 py-2 rounded text-sm ${tab === "standard" ? "bg-cyan text-white" : "bg-fg/5 text-fg-muted"}`}>Standard FL</button>
        <button onClick={() => setTab("dp")} className={`px-4 py-2 rounded text-sm ${tab === "dp" ? "bg-cyan text-white" : "bg-fg/5 text-fg-muted"}`}>DP-FedAvg</button>
      </div>

      {/* Shared params */}
      <div className="glass-card p-4 grid grid-cols-2 md:grid-cols-4 gap-3">
        <label className="text-sm text-fg-secondary">
          Clients
          <input type="number" value={nClients} onChange={(e) => setNClients(+e.target.value)} className="w-full ff-input mt-1" min={1} />
        </label>
        <label className="text-sm text-fg-secondary">
          Rounds
          <input type="number" value={nRounds} onChange={(e) => setNRounds(+e.target.value)} className="w-full ff-input mt-1" min={1} />
        </label>
        <label className="text-sm text-fg-secondary">
          Seed
          <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} className="w-full ff-input mt-1" />
        </label>
        {tab === "standard" ? (
          <label className="text-sm text-fg-secondary flex items-center gap-2 self-end">
            <input type="checkbox" checked={enablePoisoning} onChange={(e) => setEnablePoisoning(e.target.checked)} />
            Poisoning Detection
          </label>
        ) : (
          <>
            <label className="text-sm text-fg-secondary">
              Epsilon
              <input type="number" value={epsilon} onChange={(e) => setEpsilon(+e.target.value)} className="w-full ff-input mt-1" step={0.1} />
            </label>
            <label className="text-sm text-fg-secondary">
              Delta
              <input type="number" value={delta} onChange={(e) => setDelta(+e.target.value)} className="w-full ff-input mt-1" step={0.00001} />
            </label>
            <label className="text-sm text-fg-secondary">
              Max Grad Norm
              <input type="number" value={maxGradNorm} onChange={(e) => setMaxGradNorm(+e.target.value)} className="w-full ff-input mt-1" step={0.1} />
            </label>
          </>
        )}
      </div>

      <LoadingButton
        onClick={tab === "standard" ? startSimulation : startDPSimulation}
        loading={loading}
        loadingText="Running..."
      >
        Start Simulation
      </LoadingButton>

      {/* Results */}
      {(tab === "standard" ? result : dpResult) && (
        <div className="glass-card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-fg-secondary">Simulation Results</h3>
            <span className="text-xs text-fg-faint font-mono">
              {(tab === "standard" ? result : dpResult)!.simulation_id.slice(0, 12)}...
            </span>
          </div>
          {(tab === "dp" && dpResult?.total_epsilon !== undefined) && (
            <div className="mb-3 bg-gold/10 border border-gold/20 rounded-lg px-3 py-2 text-sm text-gold">
              Total privacy budget spent: epsilon = {dpResult.total_epsilon.toFixed(4)}
            </div>
          )}
          {((tab === "standard" ? result : dpResult)!.rounds).map((r) => {
            const simId = (tab === "standard" ? result : dpResult)!.simulation_id;
            return (
              <div key={r.round} className="border-b border-border-theme last:border-0 py-2">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm text-fg-secondary">Round {r.round}</span>
                  <div className="flex items-center gap-2">
                    <code className="text-xs text-fg-faint">{r.global_model_hash.slice(0, 16)}...</code>
                    <button
                      onClick={() => inspectRound(simId, r.round)}
                      className="text-xs text-cyan hover:text-cyan/80 px-2 py-0.5 rounded hover:bg-cyan/10 transition-colors"
                    >
                      Inspect
                    </button>
                  </div>
                </div>
                {r.poisoning && r.poisoning.n_flagged > 0 && (
                  <p className="text-xs text-rose-ember mt-1">Flagged: {r.poisoning.flagged_client_ids.join(", ")}</p>
                )}
                {r.noise_scale !== undefined && (
                  <p className="text-xs text-fg-muted mt-1">Noise: {r.noise_scale.toFixed(4)} | Clipped: {r.n_clipped} | Privacy: ε={r.privacy_spent?.toFixed(4)}</p>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Round Inspector */}
      {roundDetail && (
        <div className="glass-card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-fg-secondary">Round Detail</h3>
            <button onClick={() => { setRoundDetail(null); setRoundSecurity(null); }} className="text-fg-faint hover:text-fg-secondary text-sm">Close</button>
          </div>
          {Array.isArray((roundDetail as Record<string, unknown>).client_contributions) && (
            <div>
              <p className="text-xs font-semibold text-fg-muted mb-2">Client Contributions</p>
              <div className="space-y-1">
                {((roundDetail as Record<string, unknown>).client_contributions as Array<{ client_id: string; n_samples: number; local_loss: number }>).map((c) => (
                  <div key={c.client_id} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                    <span className="text-sm text-fg-secondary font-mono">{c.client_id}</span>
                    <div className="flex items-center gap-4 text-xs text-fg-muted">
                      <span>{c.n_samples} samples</span>
                      <span className={c.local_loss > 1 ? "text-rose-ember" : "text-teal"}>
                        loss: {c.local_loss.toFixed(4)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <details className="text-xs">
            <summary className="text-fg-faint cursor-pointer">Raw JSON</summary>
            <pre className="bg-surface rounded-lg p-2 overflow-x-auto mt-1">{JSON.stringify(roundDetail, null, 2)}</pre>
          </details>
        </div>
      )}

      {roundSecurity && (
        <div className="glass-card p-4 space-y-3">
          <h3 className="font-semibold text-fg-secondary">Security Analysis</h3>
          {(roundSecurity as Record<string, unknown>).poisoning_detection_enabled === false ? (
            <p className="text-sm text-fg-muted">Poisoning detection was not enabled for this simulation.</p>
          ) : (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${(roundSecurity as Record<string, unknown>).has_poisoning ? "bg-rose-ember/10 text-rose-ember" : "bg-teal/10 text-teal"}`}>
                  {(roundSecurity as Record<string, unknown>).has_poisoning ? "Poisoning Detected" : "Clean"}
                </span>
              </div>
              <details className="text-xs">
                <summary className="text-fg-faint cursor-pointer">Full report</summary>
                <pre className="bg-surface rounded-lg p-2 overflow-x-auto mt-1">{JSON.stringify(roundSecurity, null, 2)}</pre>
              </details>
            </div>
          )}
        </div>
      )}

      {/* Status */}
      {status && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-2">FL Status</h3>
          <pre className="text-xs overflow-x-auto">{JSON.stringify(status, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
