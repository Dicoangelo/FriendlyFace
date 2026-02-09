import { useState } from "react";
import { useFetch } from "../hooks/useFetch";
import LoadingButton from "../components/LoadingButton";
import EmptyState from "../components/EmptyState";

interface ExplanationRecord {
  explanation_id: string;
  method: string;
  event_id: string;
  inference_event_id: string;
  timestamp: string;
  num_superpixels?: number;
  num_samples?: number;
  top_k?: number;
  random_state?: number;
  num_regions?: number;
  computed?: boolean;
}

interface ExplanationList {
  items: ExplanationRecord[];
  total: number;
  limit: number;
  offset: number;
}

interface CompareResult {
  inference_event_id: string;
  lime_explanations: ExplanationRecord[];
  shap_explanations: ExplanationRecord[];
  sdd_explanations: ExplanationRecord[];
  total_lime: number;
  total_shap: number;
  total_sdd: number;
}

export default function Explainability() {
  const { data: explanations, error: fetchError, retry: refreshExplanations } = useFetch<ExplanationList>("/api/v1/explainability/explanations");
  const [selectedDetail, setSelectedDetail] = useState<Record<string, unknown> | null>(null);
  const [comparison, setComparison] = useState<CompareResult | null>(null);
  const [error, setError] = useState("");

  // LIME form state
  const [limeEventId, setLimeEventId] = useState("");
  const [limeSuperpixels, setLimeSuperpixels] = useState(50);
  const [limeSamples, setLimeSamples] = useState(100);
  const [limeTopK, setLimeTopK] = useState(5);
  const [limeResult, setLimeResult] = useState<Record<string, unknown> | null>(null);

  // SHAP form state
  const [shapEventId, setShapEventId] = useState("");
  const [shapSamples, setShapSamples] = useState(128);
  const [shapRandomState, setShapRandomState] = useState(42);
  const [shapResult, setShapResult] = useState<Record<string, unknown> | null>(null);

  // SDD form state
  const [sddEventId, setSddEventId] = useState("");
  const [sddResult, setSddResult] = useState<Record<string, unknown> | null>(null);

  // Compare form state
  const [compareEventId, setCompareEventId] = useState("");

  // Loading states
  const [limeLoading, setLimeLoading] = useState(false);
  const [shapLoading, setShapLoading] = useState(false);
  const [sddLoading, setSddLoading] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);

  // fetchError from useFetch is displayed below

  const triggerLime = () => {
    setError("");
    setLimeResult(null);
    if (!limeEventId.trim()) {
      setError("LIME: event_id is required");
      return;
    }
    setLimeLoading(true);
    fetch("/api/v1/explainability/lime", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        event_id: limeEventId.trim(),
        num_superpixels: limeSuperpixels,
        num_samples: limeSamples,
        top_k: limeTopK,
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setLimeResult(data);
        refreshExplanations();
      })
      .catch((e) => setError(`LIME error: ${e.message}`))
      .finally(() => setLimeLoading(false));
  };

  const triggerShap = () => {
    setError("");
    setShapResult(null);
    if (!shapEventId.trim()) {
      setError("SHAP: event_id is required");
      return;
    }
    setShapLoading(true);
    fetch("/api/v1/explainability/shap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        event_id: shapEventId.trim(),
        num_samples: shapSamples,
        random_state: shapRandomState,
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setShapResult(data);
        refreshExplanations();
      })
      .catch((e) => setError(`SHAP error: ${e.message}`))
      .finally(() => setShapLoading(false));
  };

  const triggerSdd = () => {
    setError("");
    setSddResult(null);
    if (!sddEventId.trim()) {
      setError("SDD: event_id is required");
      return;
    }
    setSddLoading(true);
    fetch("/api/v1/explainability/sdd", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: sddEventId.trim() }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setSddResult(data);
        refreshExplanations();
      })
      .catch((e) => setError(`SDD error: ${e.message}`))
      .finally(() => setSddLoading(false));
  };

  const loadComparison = () => {
    setError("");
    setComparison(null);
    if (!compareEventId.trim()) {
      setError("Compare: event_id is required");
      return;
    }
    setCompareLoading(true);
    fetch(`/api/v1/explainability/compare/${compareEventId.trim()}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setComparison)
      .catch((e) => setError(`Compare error: ${e.message}`))
      .finally(() => setCompareLoading(false));
  };

  const viewDetail = (explanationId: string) => {
    fetch(`/api/v1/explainability/explanations/${explanationId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setSelectedDetail)
      .catch((e) => setError(e.message));
  };

  const methodColors: Record<string, string> = {
    lime: "bg-teal/10 text-teal",
    shap: "bg-amethyst/10 text-amethyst",
    sdd: "bg-cyan/10 text-cyan",
  };

  return (
    <div className="space-y-6">
      {/* Page title shown in header bar */}
      {(error || fetchError) && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error || fetchError}
        </div>
      )}

      {/* Stats cards */}
      {explanations && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="glass-card p-4">
            <p className="text-sm text-fg-muted">Total Explanations</p>
            <p className="text-2xl font-bold text-fg mt-1">{explanations.total}</p>
          </div>
          <div className="glass-card p-4">
            <p className="text-sm text-fg-muted">LIME</p>
            <p className="text-2xl font-bold text-teal mt-1">
              {explanations.items.filter((e) => e.method === "lime").length}
            </p>
          </div>
          <div className="glass-card p-4">
            <p className="text-sm text-fg-muted">SHAP</p>
            <p className="text-2xl font-bold text-amethyst mt-1">
              {explanations.items.filter((e) => e.method === "shap").length}
            </p>
          </div>
          <div className="glass-card p-4">
            <p className="text-sm text-fg-muted">SDD</p>
            <p className="text-2xl font-bold text-cyan mt-1">
              {explanations.items.filter((e) => e.method === "sdd").length}
            </p>
          </div>
        </div>
      )}

      {/* Explanation list */}
      {explanations && explanations.items.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-2">Explanation History</h3>
          <table className="w-full text-sm">
            <thead className="text-left text-fg-muted border-b border-border-theme">
              <tr>
                <th className="pb-2">ID</th>
                <th className="pb-2">Method</th>
                <th className="pb-2">Status</th>
                <th className="pb-2">Inference Event</th>
                <th className="pb-2">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {explanations.items.map((e) => (
                <tr
                  key={e.explanation_id}
                  onClick={() => viewDetail(e.explanation_id)}
                  className="border-b border-border-theme hover:bg-fg/[0.02] cursor-pointer"
                >
                  <td className="py-2 font-mono text-xs">{e.explanation_id.slice(0, 8)}...</td>
                  <td className="py-2">
                    <span
                      className={`px-2 py-0.5 rounded text-xs font-medium ${methodColors[e.method] || "bg-fg/5 text-fg-secondary"}`}
                    >
                      {e.method.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-2">
                    <ComputedBadge computed={e.computed} />
                  </td>
                  <td className="py-2 font-mono text-xs text-fg-muted">
                    {e.inference_event_id.slice(0, 8)}...
                  </td>
                  <td className="py-2 text-fg-muted">{new Date(e.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Detail view */}
      {selectedDetail && (
        <div className="glass-card p-4 animate-fade-in">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-fg-secondary">Explanation Detail</h3>
              <ComputedBadge computed={(selectedDetail as Record<string, unknown>).computed as boolean | undefined} />
            </div>
            <button
              onClick={() => setSelectedDetail(null)}
              className="text-fg-faint hover:text-fg-secondary text-sm"
            >
              Close
            </button>
          </div>
          <RichDetailView detail={selectedDetail} />
        </div>
      )}

      {explanations && explanations.items.length === 0 && (
        <div className="glass-card">
          <EmptyState
            title="No explanations generated yet"
            subtitle="Use the forms below to generate LIME, SHAP, or SDD explanations for inference events"
          />
        </div>
      )}

      {/* Trigger LIME */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Generate LIME Explanation</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Event ID
            <input
              type="text"
              value={limeEventId}
              onChange={(e) => setLimeEventId(e.target.value)}
              className="ff-input font-mono text-xs w-72 block mt-1"
              placeholder="Inference event UUID"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Superpixels
            <input
              type="number"
              value={limeSuperpixels}
              onChange={(e) => setLimeSuperpixels(+e.target.value)}
              className="ff-input w-20 block mt-1"
              min={4}
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Samples
            <input
              type="number"
              value={limeSamples}
              onChange={(e) => setLimeSamples(+e.target.value)}
              className="ff-input w-20 block mt-1"
              min={10}
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Top K
            <input
              type="number"
              value={limeTopK}
              onChange={(e) => setLimeTopK(+e.target.value)}
              className="ff-input w-20 block mt-1"
              min={1}
            />
          </label>
        </div>
        <LoadingButton onClick={triggerLime} loading={limeLoading} className="btn-success" loadingText="Running...">
          Run LIME
        </LoadingButton>
        {limeResult && (
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
            {JSON.stringify(limeResult, null, 2)}
          </pre>
        )}
      </div>

      {/* Trigger SHAP */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Generate SHAP Explanation</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Event ID
            <input
              type="text"
              value={shapEventId}
              onChange={(e) => setShapEventId(e.target.value)}
              className="ff-input font-mono text-xs w-72 block mt-1"
              placeholder="Inference event UUID"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Samples
            <input
              type="number"
              value={shapSamples}
              onChange={(e) => setShapSamples(+e.target.value)}
              className="ff-input w-20 block mt-1"
              min={10}
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Random State
            <input
              type="number"
              value={shapRandomState}
              onChange={(e) => setShapRandomState(+e.target.value)}
              className="ff-input w-20 block mt-1"
            />
          </label>
        </div>
        <LoadingButton onClick={triggerShap} loading={shapLoading} className="btn-accent" loadingText="Running...">
          Run SHAP
        </LoadingButton>
        {shapResult && (
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
            {JSON.stringify(shapResult, null, 2)}
          </pre>
        )}
      </div>

      {/* Trigger SDD */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Generate SDD Explanation</h3>
        <div className="flex gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Event ID
            <input
              type="text"
              value={sddEventId}
              onChange={(e) => setSddEventId(e.target.value)}
              className="ff-input font-mono text-xs w-72 block mt-1"
              placeholder="Inference event UUID"
            />
          </label>
        </div>
        <LoadingButton onClick={triggerSdd} loading={sddLoading} loadingText="Running...">
          Run SDD
        </LoadingButton>
        {sddResult && (
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
            {JSON.stringify(sddResult, null, 2)}
          </pre>
        )}
      </div>

      {/* Compare explanations */}
      <div className="glass-card p-4 space-y-2">
        <h3 className="font-semibold text-fg-secondary">Compare Explanations</h3>
        <p className="text-xs text-fg-muted">
          Compare LIME, SHAP, and SDD explanations side-by-side for the same inference event.
        </p>
        <div className="flex gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Inference Event ID
            <input
              type="text"
              value={compareEventId}
              onChange={(e) => setCompareEventId(e.target.value)}
              className="ff-input font-mono text-xs w-72 block mt-1"
              placeholder="Inference event UUID"
            />
          </label>
          <LoadingButton onClick={loadComparison} loading={compareLoading} loadingText="Comparing...">
            Compare
          </LoadingButton>
        </div>
      </div>

      {/* Comparison results */}
      {comparison && (
        <div className="glass-card p-4 animate-fade-in">
          <h3 className="font-semibold text-fg-secondary mb-3">
            Comparison for{" "}
            <span className="font-mono text-xs">{comparison.inference_event_id.slice(0, 12)}...</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <CompareColumn
              title="LIME"
              count={comparison.total_lime}
              items={comparison.lime_explanations}
              color="teal"
            />
            <CompareColumn
              title="SHAP"
              count={comparison.total_shap}
              items={comparison.shap_explanations}
              color="amethyst"
            />
            <CompareColumn
              title="SDD"
              count={comparison.total_sdd}
              items={comparison.sdd_explanations}
              color="cyan"
            />
          </div>
        </div>
      )}
    </div>
  );
}

function ComputedBadge({ computed }: { computed?: boolean }) {
  if (computed === true) {
    return (
      <span className="px-2 py-0.5 rounded text-xs font-medium bg-teal/10 text-teal border border-teal/20">
        Computed
      </span>
    );
  }
  return (
    <span className="px-2 py-0.5 rounded text-xs font-medium bg-fg/5 text-fg-faint border border-fg-faint/20">
      Stub
    </span>
  );
}

function RichDetailView({ detail }: { detail: Record<string, unknown> }) {
  const computed = detail.computed === true;
  const method = (detail.method as string) || "";

  if (!computed) {
    return (
      <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
        {JSON.stringify(detail, null, 2)}
      </pre>
    );
  }

  // LIME rich view
  if (method === "lime" && detail.top_regions) {
    const regions = detail.top_regions as Array<{ region: number; weight: number }>;
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-teal">LIME Top Regions</h4>
        <div className="space-y-1">
          {regions.map((r, i) => {
            const maxWeight = Math.max(...regions.map((x) => Math.abs(x.weight)));
            const pct = maxWeight > 0 ? (Math.abs(r.weight) / maxWeight) * 100 : 0;
            return (
              <div key={i} className="flex items-center gap-3">
                <span className="text-xs text-fg-muted w-24">Region {r.region}</span>
                <div className="flex-1 h-5 bg-surface rounded-md overflow-hidden relative">
                  <div
                    className={`h-full rounded-md ${r.weight >= 0 ? "bg-teal/30" : "bg-rose-ember/30"} transition-all`}
                    style={{ width: `${pct}%` }}
                  />
                  <span className={`absolute inset-y-0 right-2 flex items-center text-xs font-semibold ${r.weight >= 0 ? "text-teal" : "text-rose-ember"}`}>
                    {r.weight.toFixed(4)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
        <details className="text-xs">
          <summary className="text-fg-faint cursor-pointer">Raw JSON</summary>
          <pre className="bg-surface rounded-lg p-2 overflow-x-auto mt-1">
            {JSON.stringify(detail, null, 2)}
          </pre>
        </details>
      </div>
    );
  }

  // SHAP rich view
  if (method === "shap" && detail.top_features) {
    const features = detail.top_features as Array<{ feature: string; value: number }>;
    const baseValue = detail.base_value as number | undefined;
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-amethyst">SHAP Feature Attributions</h4>
        {baseValue !== undefined && (
          <p className="text-xs text-fg-muted">Base value: <span className="font-mono text-fg-secondary">{baseValue.toFixed(4)}</span></p>
        )}
        <div className="space-y-1">
          {features.map((f, i) => {
            const maxVal = Math.max(...features.map((x) => Math.abs(x.value)));
            const pct = maxVal > 0 ? (Math.abs(f.value) / maxVal) * 100 : 0;
            return (
              <div key={i} className="flex items-center gap-3">
                <span className="text-xs text-fg-muted w-24 truncate">{f.feature}</span>
                <div className="flex-1 h-5 bg-surface rounded-md overflow-hidden relative">
                  <div
                    className={`h-full rounded-md ${f.value >= 0 ? "bg-amethyst/30" : "bg-rose-ember/30"} transition-all`}
                    style={{ width: `${pct}%` }}
                  />
                  <span className={`absolute inset-y-0 right-2 flex items-center text-xs font-semibold ${f.value >= 0 ? "text-amethyst" : "text-rose-ember"}`}>
                    {f.value.toFixed(4)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
        <details className="text-xs">
          <summary className="text-fg-faint cursor-pointer">Raw JSON</summary>
          <pre className="bg-surface rounded-lg p-2 overflow-x-auto mt-1">
            {JSON.stringify(detail, null, 2)}
          </pre>
        </details>
      </div>
    );
  }

  // SDD rich view
  if (method === "sdd" && detail.regions) {
    const regions = detail.regions as Array<{ region: string; saliency: number }>;
    const dominant = detail.dominant_region as string | undefined;
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-cyan">SDD Saliency Regions</h4>
        {dominant && (
          <p className="text-xs text-fg-muted">Dominant region: <span className="font-mono text-cyan">{dominant}</span></p>
        )}
        <div className="space-y-1">
          {regions.map((r, i) => {
            const maxSal = Math.max(...regions.map((x) => x.saliency));
            const pct = maxSal > 0 ? (r.saliency / maxSal) * 100 : 0;
            return (
              <div key={i} className="flex items-center gap-3">
                <span className="text-xs text-fg-muted w-24 truncate">{r.region}</span>
                <div className="flex-1 h-5 bg-surface rounded-md overflow-hidden relative">
                  <div
                    className="h-full rounded-md bg-cyan/30 transition-all"
                    style={{ width: `${pct}%` }}
                  />
                  <span className="absolute inset-y-0 right-2 flex items-center text-xs font-semibold text-cyan">
                    {r.saliency.toFixed(4)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
        <details className="text-xs">
          <summary className="text-fg-faint cursor-pointer">Raw JSON</summary>
          <pre className="bg-surface rounded-lg p-2 overflow-x-auto mt-1">
            {JSON.stringify(detail, null, 2)}
          </pre>
        </details>
      </div>
    );
  }

  // Fallback for computed but unknown structure
  return (
    <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
      {JSON.stringify(detail, null, 2)}
    </pre>
  );
}

function CompareColumn({
  title,
  count,
  items,
  color,
}: {
  title: string;
  count: number;
  items: ExplanationRecord[];
  color: string;
}) {
  const colorMap: Record<string, { border: string; bg: string; text: string }> = {
    teal: { border: "border-teal/20", bg: "bg-teal/5", text: "text-teal" },
    amethyst: { border: "border-amethyst/20", bg: "bg-amethyst/5", text: "text-amethyst" },
    cyan: { border: "border-cyan/20", bg: "bg-cyan/5", text: "text-cyan" },
  };

  const colors = colorMap[color] || colorMap.cyan;

  return (
    <div className={`rounded-lg border-2 ${colors.border} ${colors.bg} p-3`}>
      <div className="flex items-center justify-between mb-2">
        <span className={`font-semibold ${colors.text}`}>{title}</span>
        <span className="text-xs text-fg-muted">{count} result{count !== 1 ? "s" : ""}</span>
      </div>
      {items.length === 0 ? (
        <p className="text-xs text-fg-faint">No explanations</p>
      ) : (
        <div className="space-y-2">
          {items.map((item) => (
            <div key={item.explanation_id} className="glass-card p-2 text-xs">
              <p className="font-mono text-fg-secondary">{item.explanation_id.slice(0, 12)}...</p>
              <p className="text-fg-faint">{new Date(item.timestamp).toLocaleString()}</p>
              {item.num_samples !== undefined && <p>Samples: {item.num_samples}</p>}
              {item.num_superpixels !== undefined && <p>Superpixels: {item.num_superpixels}</p>}
              {item.num_regions !== undefined && <p>Regions: {item.num_regions}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
