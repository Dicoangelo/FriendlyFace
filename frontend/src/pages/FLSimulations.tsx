import { useState } from "react";

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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Federated Learning</h2>
        <button onClick={fetchStatus} className="px-3 py-1 border rounded text-sm hover:bg-gray-50">Refresh Status</button>
      </div>
      {error && <div className="text-red-600 text-sm">{error}</div>}

      {/* Tabs */}
      <div className="flex gap-2">
        <button onClick={() => setTab("standard")} className={`px-4 py-2 rounded text-sm ${tab === "standard" ? "bg-blue-600 text-white" : "bg-gray-200"}`}>Standard FL</button>
        <button onClick={() => setTab("dp")} className={`px-4 py-2 rounded text-sm ${tab === "dp" ? "bg-blue-600 text-white" : "bg-gray-200"}`}>DP-FedAvg</button>
      </div>

      {/* Shared params */}
      <div className="bg-white rounded-lg shadow p-4 grid grid-cols-2 md:grid-cols-4 gap-3">
        <label className="text-sm">
          Clients
          <input type="number" value={nClients} onChange={(e) => setNClients(+e.target.value)} className="w-full border rounded px-2 py-1 mt-1" min={1} />
        </label>
        <label className="text-sm">
          Rounds
          <input type="number" value={nRounds} onChange={(e) => setNRounds(+e.target.value)} className="w-full border rounded px-2 py-1 mt-1" min={1} />
        </label>
        <label className="text-sm">
          Seed
          <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} className="w-full border rounded px-2 py-1 mt-1" />
        </label>
        {tab === "standard" ? (
          <label className="text-sm flex items-center gap-2 self-end">
            <input type="checkbox" checked={enablePoisoning} onChange={(e) => setEnablePoisoning(e.target.checked)} />
            Poisoning Detection
          </label>
        ) : (
          <>
            <label className="text-sm">
              Epsilon
              <input type="number" value={epsilon} onChange={(e) => setEpsilon(+e.target.value)} className="w-full border rounded px-2 py-1 mt-1" step={0.1} />
            </label>
            <label className="text-sm">
              Delta
              <input type="number" value={delta} onChange={(e) => setDelta(+e.target.value)} className="w-full border rounded px-2 py-1 mt-1" step={0.00001} />
            </label>
            <label className="text-sm">
              Max Grad Norm
              <input type="number" value={maxGradNorm} onChange={(e) => setMaxGradNorm(+e.target.value)} className="w-full border rounded px-2 py-1 mt-1" step={0.1} />
            </label>
          </>
        )}
      </div>

      <button onClick={tab === "standard" ? startSimulation : startDPSimulation} disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50">
        {loading ? "Running..." : "Start Simulation"}
      </button>

      {/* Results */}
      {(tab === "standard" ? result : dpResult) && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold mb-2">Simulation Results</h3>
          {((tab === "standard" ? result : dpResult)!.rounds).map((r) => (
            <div key={r.round} className="border-b last:border-0 py-2">
              <div className="flex items-center justify-between">
                <span className="font-medium text-sm">Round {r.round}</span>
                <code className="text-xs text-gray-400">{r.global_model_hash.slice(0, 16)}...</code>
              </div>
              {r.poisoning && r.poisoning.n_flagged > 0 && (
                <p className="text-xs text-red-600 mt-1">Flagged: {r.poisoning.flagged_client_ids.join(", ")}</p>
              )}
              {r.noise_scale !== undefined && (
                <p className="text-xs text-gray-500 mt-1">Noise: {r.noise_scale.toFixed(4)} | Clipped: {r.n_clipped} | Privacy: ε={r.privacy_spent?.toFixed(4)}</p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Status */}
      {status && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold mb-2">FL Status</h3>
          <pre className="text-xs overflow-x-auto">{JSON.stringify(status, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
