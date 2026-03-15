import { useEffect, useState } from "react";
import LoadingButton from "../components/LoadingButton";
import { SkeletonCard } from "../components/Skeleton";

interface SimRound {
  round: number;
  global_model_hash: string;
  event_id: string;
  recorded_event_id?: string;
  poisoning?: { flagged_client_ids: string[]; n_flagged: number };
  noise_scale?: number;
  n_clipped?: number;
  clipped_clients?: string[];
  privacy_spent?: number;
}

interface SimResult {
  simulation_id: string;
  n_rounds: number;
  n_clients: number;
  final_model_hash?: string;
  mode?: string;
  rounds: SimRound[];
  dp_config?: { epsilon: number; delta: number; max_grad_norm: number };
  total_epsilon?: number;
}

export default function FLSimulations() {
  const [tab, setTab] = useState<"standard" | "dp">("standard");

  // Shared params
  const [nClients, setNClients] = useState(5);
  const [nRounds, setNRounds] = useState(3);
  const [enablePoisoning, setEnablePoisoning] = useState(true);
  const [seed, setSeed] = useState(42);

  // DP params
  const [epsilon, setEpsilon] = useState(1.0);
  const [delta, setDelta] = useState(1e-5);
  const [maxGradNorm, setMaxGradNorm] = useState(1.0);

  // Results
  const [result, setResult] = useState<SimResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // FL Status
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = () => {
    setStatusLoading(true);
    fetch("/api/v1/fl/status")
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setStatus(data);
        setStatusLoading(false);
      })
      .catch((e) => {
        setError(`FL status error: ${e.message}`);
        setStatusLoading(false);
      });
  };

  const runSimulation = () => {
    setLoading(true);
    setError("");
    setResult(null);

    const url = tab === "standard" ? "/api/v1/fl/start" : "/api/v1/fl/dp-start";
    const body =
      tab === "standard"
        ? { n_clients: nClients, n_rounds: nRounds, enable_poisoning_detection: enablePoisoning, seed }
        : { n_clients: nClients, n_rounds: nRounds, epsilon, delta, max_grad_norm: maxGradNorm, seed };

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setResult(data);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  };

  const activeResult = result;

  return (
    <div className="space-y-6">
      {/* Error banner */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-3 text-rose-ember text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError("")} className="text-rose-ember/60 hover:text-rose-ember ml-4">
            &times;
          </button>
        </div>
      )}

      {/* Pill tabs */}
      <div className="inline-flex bg-surface rounded-full p-1">
        <button
          onClick={() => setTab("standard")}
          className={`px-5 py-1.5 rounded-full text-sm font-medium transition-colors ${
            tab === "standard" ? "bg-cyan text-white" : "text-fg-muted hover:text-fg-secondary"
          }`}
        >
          Standard FL
        </button>
        <button
          onClick={() => setTab("dp")}
          className={`px-5 py-1.5 rounded-full text-sm font-medium transition-colors ${
            tab === "dp" ? "bg-cyan text-white" : "text-fg-muted hover:text-fg-secondary"
          }`}
        >
          DP-FedAvg
        </button>
      </div>

      {/* Parameters form */}
      <div className="glass-card p-4">
        <h3 className="text-sm font-semibold text-fg-secondary mb-3">
          {tab === "standard" ? "Standard FL Parameters" : "DP-FedAvg Parameters"}
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <label className="text-sm text-fg-secondary">
            Clients
            <input
              type="number"
              value={nClients}
              onChange={(e) => setNClients(Math.max(2, Math.min(20, +e.target.value)))}
              className="w-full ff-input mt-1"
              min={2}
              max={20}
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Rounds
            <input
              type="number"
              value={nRounds}
              onChange={(e) => setNRounds(Math.max(1, Math.min(10, +e.target.value)))}
              className="w-full ff-input mt-1"
              min={1}
              max={10}
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(+e.target.value)}
              className="w-full ff-input mt-1"
            />
          </label>
          {tab === "standard" ? (
            <label className="text-sm text-fg-secondary flex items-center gap-2 self-end pb-2">
              <input
                type="checkbox"
                checked={enablePoisoning}
                onChange={(e) => setEnablePoisoning(e.target.checked)}
                className="accent-amethyst"
              />
              Poisoning Detection
            </label>
          ) : (
            <>
              <label className="text-sm text-fg-secondary">
                <span className="flex justify-between">
                  Epsilon
                  <span className="text-fg-faint">{epsilon.toFixed(1)}</span>
                </span>
                <input
                  type="range"
                  value={epsilon}
                  onChange={(e) => setEpsilon(+e.target.value)}
                  className="w-full mt-1 accent-amethyst"
                  min={0.1}
                  max={10}
                  step={0.1}
                />
              </label>
              <label className="text-sm text-fg-secondary">
                Delta
                <input
                  type="number"
                  value={delta}
                  onChange={(e) => setDelta(+e.target.value)}
                  className="w-full ff-input mt-1"
                  step={0.00001}
                  min={0}
                />
              </label>
              <label className="text-sm text-fg-secondary">
                Max Grad Norm
                <input
                  type="number"
                  value={maxGradNorm}
                  onChange={(e) => setMaxGradNorm(+e.target.value)}
                  className="w-full ff-input mt-1"
                  step={0.1}
                  min={0.1}
                />
              </label>
            </>
          )}
        </div>
        <div className="mt-4">
          <LoadingButton onClick={runSimulation} loading={loading} loadingText="Running...">
            Run Simulation
          </LoadingButton>
        </div>
      </div>

      {/* Results */}
      {activeResult && (
        <>
          {/* Summary card */}
          <div className="glass-card p-4">
            <h3 className="text-sm font-semibold text-fg-secondary mb-3">Simulation Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-fg-faint">Simulation ID</span>
                <p className="font-mono text-fg-secondary truncate">{activeResult.simulation_id}</p>
              </div>
              <div>
                <span className="text-fg-faint">Total Rounds</span>
                <p className="text-fg-secondary font-medium">{activeResult.n_rounds}</p>
              </div>
              <div>
                <span className="text-fg-faint">Final Model Hash</span>
                <p className="font-mono text-fg-secondary truncate">
                  {activeResult.final_model_hash
                    ? activeResult.final_model_hash.slice(0, 16) + "..."
                    : activeResult.rounds[activeResult.rounds.length - 1]?.global_model_hash.slice(0, 16) + "..."}
                </p>
              </div>
              {activeResult.dp_config && (
                <div>
                  <span className="text-fg-faint">DP Budget Spent</span>
                  <p className="text-fg-secondary font-medium">
                    &epsilon; = {activeResult.total_epsilon?.toFixed(4)}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Rounds timeline */}
          <div className="glass-card p-4">
            <h3 className="text-sm font-semibold text-fg-secondary mb-3">Round Timeline</h3>
            <div className="space-y-0">
              {activeResult.rounds.map((r) => {
                const progress = (r.round / activeResult.n_rounds) * 100;
                return (
                  <div key={r.round} className="border-b border-border-theme last:border-0 py-3">
                    <div className="flex items-center gap-4">
                      <span className="text-sm font-medium text-fg-secondary w-20 shrink-0">
                        Round {r.round}
                      </span>
                      <div className="flex-1">
                        <div className="w-full bg-surface rounded-full h-2">
                          <div
                            className="bg-cyan h-2 rounded-full transition-all"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                      </div>
                      <code className="text-xs text-fg-faint shrink-0">
                        {r.global_model_hash.slice(0, 16)}...
                      </code>
                    </div>
                    {r.poisoning && r.poisoning.n_flagged > 0 && (
                      <div className="mt-1.5 ml-20 flex items-center gap-1.5">
                        <span className="px-2 py-0.5 rounded text-xs font-medium bg-rose-ember/10 text-rose-ember">
                          Poisoning Flagged
                        </span>
                        <span className="text-xs text-fg-muted">
                          Clients: {r.poisoning.flagged_client_ids.join(", ")}
                        </span>
                      </div>
                    )}
                    {r.noise_scale !== undefined && (
                      <div className="mt-1.5 ml-20 text-xs text-fg-muted flex gap-3">
                        <span>Noise: {r.noise_scale.toFixed(4)}</span>
                        <span>Clipped: {r.n_clipped}</span>
                        <span>&epsilon; = {r.privacy_spent?.toFixed(4)}</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}

      {/* FL Status card */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-fg-secondary">FL Status</h3>
          <button onClick={fetchStatus} className="text-xs text-cyan hover:text-cyan/80 transition-colors">
            Refresh
          </button>
        </div>
        {statusLoading ? (
          <SkeletonCard className="h-16" />
        ) : status ? (
          <pre className="text-xs overflow-x-auto bg-surface rounded-lg p-3">{JSON.stringify(status, null, 2)}</pre>
        ) : (
          <p className="text-sm text-fg-muted">Unable to load FL status.</p>
        )}
      </div>
    </div>
  );
}
