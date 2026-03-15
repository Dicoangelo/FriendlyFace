import { useState } from "react";
import LoadingButton from "../components/LoadingButton";
import { SkeletonCard } from "../components/Skeleton";
import { useCopyToClipboard } from "../hooks/useCopyToClipboard";

type Tab = "lime" | "shap" | "sdd";

interface LimeRegion {
  region: number;
  weight: number;
}

interface LimeResult {
  explanation_id: string;
  method: string;
  top_regions?: LimeRegion[];
  computed?: boolean;
  [key: string]: unknown;
}

interface ShapFeature {
  feature: string;
  value: number;
}

interface ShapResult {
  explanation_id: string;
  method: string;
  base_value?: number;
  top_features?: ShapFeature[];
  computed?: boolean;
  [key: string]: unknown;
}

interface SddResult {
  explanation_id: string;
  method: string;
  dominant_region?: string;
  regions?: Array<{ region: string; saliency: number }>;
  gradient_magnitude?: number;
  class_label?: string;
  confidence?: number;
  computed?: boolean;
  [key: string]: unknown;
}

function ErrorBanner({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-3 text-rose-ember text-sm flex items-center justify-between">
      <span>{message}</span>
      <button onClick={onDismiss} className="text-rose-ember/60 hover:text-rose-ember ml-4">&times;</button>
    </div>
  );
}

function CopyJsonButton({ data }: { data: unknown }) {
  const copy = useCopyToClipboard();
  return (
    <button
      onClick={() => copy(JSON.stringify(data, null, 2), "JSON copied")}
      className="text-fg-faint hover:text-fg text-xs px-2 py-1 rounded hover:bg-fg/5 transition-colors"
    >
      Copy JSON
    </button>
  );
}

function LimeSection() {
  const [eventId, setEventId] = useState("");
  const [superpixels, setSuperpixels] = useState(50);
  const [samples, setSamples] = useState(100);
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<LimeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generate = () => {
    setError("");
    setResult(null);
    if (!eventId.trim()) { setError("Event ID is required"); return; }
    setLoading(true);
    fetch("/api/v1/explainability/lime", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        event_id: eventId.trim(),
        num_superpixels: superpixels,
        num_samples: samples,
        top_k: topK,
      }),
    })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setResult)
      .catch((e) => setError(`LIME error: ${e.message}`))
      .finally(() => setLoading(false));
  };

  const regions = result?.top_regions ?? [];
  const maxWeight = regions.length > 0 ? Math.max(...regions.map((r) => Math.abs(r.weight))) : 0;

  return (
    <div className="space-y-4">
      {error && <ErrorBanner message={error} onDismiss={() => setError("")} />}

      <div className="flex flex-wrap gap-4 items-end">
        <label className="text-sm text-fg-secondary">
          Event ID
          <input
            type="text"
            value={eventId}
            onChange={(e) => setEventId(e.target.value)}
            className="ff-input font-mono text-xs w-72 block mt-1"
            placeholder="Inference event UUID"
          />
        </label>
        <label className="text-sm text-fg-secondary">
          Superpixels
          <input type="number" value={superpixels} onChange={(e) => setSuperpixels(+e.target.value)}
            className="ff-input w-20 block mt-1" min={4} />
        </label>
        <label className="text-sm text-fg-secondary">
          Samples
          <input type="number" value={samples} onChange={(e) => setSamples(+e.target.value)}
            className="ff-input w-20 block mt-1" min={10} />
        </label>
        <label className="text-sm text-fg-secondary">
          Top K
          <input type="number" value={topK} onChange={(e) => setTopK(+e.target.value)}
            className="ff-input w-20 block mt-1" min={1} />
        </label>
      </div>

      <LoadingButton onClick={generate} loading={loading} className="btn-success" loadingText="Generating...">
        Generate LIME
      </LoadingButton>

      {loading && <SkeletonCard className="h-32" />}

      {result && !loading && (
        <div className="space-y-3 animate-fade-in">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-teal">Feature Importance</h4>
            <CopyJsonButton data={result} />
          </div>
          {regions.length > 0 ? (
            <div className="space-y-1">
              {regions.map((r, i) => {
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
          ) : (
            <pre className="text-xs bg-surface rounded-lg p-3 overflow-x-auto">{JSON.stringify(result, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  );
}

function ShapSection() {
  const [eventId, setEventId] = useState("");
  const [samples, setSamples] = useState(128);
  const [randomState, setRandomState] = useState(42);
  const [result, setResult] = useState<ShapResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generate = () => {
    setError("");
    setResult(null);
    if (!eventId.trim()) { setError("Event ID is required"); return; }
    setLoading(true);
    fetch("/api/v1/explainability/shap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        event_id: eventId.trim(),
        num_samples: samples,
        random_state: randomState,
      }),
    })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setResult)
      .catch((e) => setError(`SHAP error: ${e.message}`))
      .finally(() => setLoading(false));
  };

  const features = result?.top_features ?? [];
  const maxVal = features.length > 0 ? Math.max(...features.map((f) => Math.abs(f.value))) : 0;

  return (
    <div className="space-y-4">
      {error && <ErrorBanner message={error} onDismiss={() => setError("")} />}

      <div className="flex flex-wrap gap-4 items-end">
        <label className="text-sm text-fg-secondary">
          Event ID
          <input
            type="text"
            value={eventId}
            onChange={(e) => setEventId(e.target.value)}
            className="ff-input font-mono text-xs w-72 block mt-1"
            placeholder="Inference event UUID"
          />
        </label>
        <label className="text-sm text-fg-secondary">
          Samples
          <input type="number" value={samples} onChange={(e) => setSamples(+e.target.value)}
            className="ff-input w-20 block mt-1" min={10} />
        </label>
        <label className="text-sm text-fg-secondary">
          Random State
          <input type="number" value={randomState} onChange={(e) => setRandomState(+e.target.value)}
            className="ff-input w-20 block mt-1" />
        </label>
      </div>

      <LoadingButton onClick={generate} loading={loading} className="btn-accent" loadingText="Generating...">
        Generate SHAP
      </LoadingButton>

      {loading && <SkeletonCard className="h-32" />}

      {result && !loading && (
        <div className="space-y-3 animate-fade-in">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-amethyst">Feature Contributions</h4>
            <CopyJsonButton data={result} />
          </div>
          {result.base_value !== undefined && (
            <p className="text-xs text-fg-muted">
              Base value: <span className="font-mono text-fg-secondary">{result.base_value.toFixed(4)}</span>
            </p>
          )}
          {features.length > 0 ? (
            <div className="space-y-1">
              {features.map((f, i) => {
                const pct = maxVal > 0 ? (Math.abs(f.value) / maxVal) * 100 : 0;
                const positive = f.value >= 0;
                return (
                  <div key={i} className="flex items-center gap-3">
                    <span className="text-xs text-fg-muted w-24 truncate">{f.feature}</span>
                    <div className="flex-1 h-5 bg-surface rounded-md overflow-hidden relative">
                      <div
                        className={`h-full rounded-md ${positive ? "bg-teal/30" : "bg-rose-ember/30"} transition-all`}
                        style={{ width: `${pct}%` }}
                      />
                      <span className={`absolute inset-y-0 right-2 flex items-center text-xs font-semibold ${positive ? "text-teal" : "text-rose-ember"}`}>
                        {positive ? "+" : ""}{f.value.toFixed(4)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <pre className="text-xs bg-surface rounded-lg p-3 overflow-x-auto">{JSON.stringify(result, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  );
}

function SddSection() {
  const [eventId, setEventId] = useState("");
  const [result, setResult] = useState<SddResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generate = () => {
    setError("");
    setResult(null);
    if (!eventId.trim()) { setError("Event ID is required"); return; }
    setLoading(true);
    fetch("/api/v1/explainability/sdd", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: eventId.trim() }),
    })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setResult)
      .catch((e) => setError(`SDD error: ${e.message}`))
      .finally(() => setLoading(false));
  };

  return (
    <div className="space-y-4">
      {error && <ErrorBanner message={error} onDismiss={() => setError("")} />}

      <div className="flex gap-4 items-end">
        <label className="text-sm text-fg-secondary">
          Event ID
          <input
            type="text"
            value={eventId}
            onChange={(e) => setEventId(e.target.value)}
            className="ff-input font-mono text-xs w-72 block mt-1"
            placeholder="Inference event UUID"
          />
        </label>
      </div>

      <LoadingButton onClick={generate} loading={loading} loadingText="Generating...">
        Generate SDD
      </LoadingButton>

      {loading && <SkeletonCard className="h-32" />}

      {result && !loading && (
        <div className="space-y-3 animate-fade-in">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-cyan">SDD Saliency</h4>
            <CopyJsonButton data={result} />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {result.gradient_magnitude !== undefined && (
              <div className="bg-surface rounded-lg p-3">
                <p className="text-[11px] uppercase tracking-wider text-fg-faint">Gradient Magnitude</p>
                <p className="text-lg font-bold text-cyan mt-1">{result.gradient_magnitude.toFixed(4)}</p>
              </div>
            )}
            {result.class_label !== undefined && (
              <div className="bg-surface rounded-lg p-3">
                <p className="text-[11px] uppercase tracking-wider text-fg-faint">Class Label</p>
                <p className="text-lg font-bold text-fg mt-1">{result.class_label}</p>
              </div>
            )}
            {result.confidence !== undefined && (
              <div className="bg-surface rounded-lg p-3">
                <p className="text-[11px] uppercase tracking-wider text-fg-faint">Confidence</p>
                <p className="text-lg font-bold text-teal mt-1">{(result.confidence * 100).toFixed(1)}%</p>
              </div>
            )}
          </div>

          {result.dominant_region && (
            <p className="text-xs text-fg-muted">
              Dominant region: <span className="font-mono text-cyan">{result.dominant_region}</span>
            </p>
          )}

          {result.regions && result.regions.length > 0 && (
            <div className="space-y-1">
              {result.regions.map((r, i) => {
                const maxSal = Math.max(...(result.regions ?? []).map((x) => x.saliency));
                const pct = maxSal > 0 ? (r.saliency / maxSal) * 100 : 0;
                return (
                  <div key={i} className="flex items-center gap-3">
                    <span className="text-xs text-fg-muted w-24 truncate">{r.region}</span>
                    <div className="flex-1 h-5 bg-surface rounded-md overflow-hidden relative">
                      <div className="h-full rounded-md bg-cyan/30 transition-all" style={{ width: `${pct}%` }} />
                      <span className="absolute inset-y-0 right-2 flex items-center text-xs font-semibold text-cyan">
                        {r.saliency.toFixed(4)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {!result.regions && !result.gradient_magnitude && (
            <pre className="text-xs bg-surface rounded-lg p-3 overflow-x-auto">{JSON.stringify(result, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function Explainability() {
  const [activeTab, setActiveTab] = useState<Tab>("lime");

  const tabs: { key: Tab; label: string; color: string }[] = [
    { key: "lime", label: "LIME", color: "teal" },
    { key: "shap", label: "SHAP", color: "amethyst" },
    { key: "sdd", label: "SDD", color: "cyan" },
  ];

  return (
    <div className="space-y-6">
      {/* Tab switcher */}
      <div className="inline-flex bg-surface rounded-full p-1">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-5 py-1.5 rounded-full text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? `bg-${tab.color} text-white`
                : "text-fg-muted hover:text-fg-secondary"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="glass-card p-5">
        <h3 className="font-semibold text-fg-secondary mb-4">
          {activeTab === "lime" && "LIME — Local Interpretable Model-agnostic Explanations"}
          {activeTab === "shap" && "SHAP — SHapley Additive exPlanations"}
          {activeTab === "sdd" && "SDD — Saliency-Driven Decomposition"}
        </h3>

        {activeTab === "lime" && <LimeSection />}
        {activeTab === "shap" && <ShapSection />}
        {activeTab === "sdd" && <SddSection />}
      </div>
    </div>
  );
}
