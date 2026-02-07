import { useEffect, useState } from "react";

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
  const [explanations, setExplanations] = useState<ExplanationList | null>(null);
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

  // Compare form state
  const [compareEventId, setCompareEventId] = useState("");

  const fetchExplanations = () => {
    fetch("/api/v1/explainability/explanations")
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setExplanations)
      .catch((e) => setError(e.message));
  };

  useEffect(() => {
    fetchExplanations();
  }, []);

  const triggerLime = () => {
    setError("");
    setLimeResult(null);
    if (!limeEventId.trim()) {
      setError("LIME: event_id is required");
      return;
    }
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
        fetchExplanations();
      })
      .catch((e) => setError(`LIME error: ${e.message}`));
  };

  const triggerShap = () => {
    setError("");
    setShapResult(null);
    if (!shapEventId.trim()) {
      setError("SHAP: event_id is required");
      return;
    }
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
        fetchExplanations();
      })
      .catch((e) => setError(`SHAP error: ${e.message}`));
  };

  const loadComparison = () => {
    setError("");
    setComparison(null);
    if (!compareEventId.trim()) {
      setError("Compare: event_id is required");
      return;
    }
    fetch(`/api/v1/explainability/compare/${compareEventId.trim()}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setComparison)
      .catch((e) => setError(`Compare error: ${e.message}`));
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
    lime: "bg-green-100 text-green-700",
    shap: "bg-purple-100 text-purple-700",
    sdd: "bg-blue-100 text-blue-700",
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Explainability</h2>
      {error && <div className="text-red-600 text-sm">{error}</div>}

      {/* Stats cards */}
      {explanations && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">Total Explanations</p>
            <p className="text-2xl font-bold text-gray-800 mt-1">{explanations.total}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">LIME</p>
            <p className="text-2xl font-bold text-green-700 mt-1">
              {explanations.items.filter((e) => e.method === "lime").length}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">SHAP</p>
            <p className="text-2xl font-bold text-purple-700 mt-1">
              {explanations.items.filter((e) => e.method === "shap").length}
            </p>
          </div>
        </div>
      )}

      {/* Explanation list */}
      {explanations && explanations.items.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold mb-2">Explanation History</h3>
          <table className="w-full text-sm">
            <thead className="text-left text-gray-500 border-b">
              <tr>
                <th className="pb-2">ID</th>
                <th className="pb-2">Method</th>
                <th className="pb-2">Inference Event</th>
                <th className="pb-2">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {explanations.items.map((e) => (
                <tr
                  key={e.explanation_id}
                  onClick={() => viewDetail(e.explanation_id)}
                  className="border-b hover:bg-gray-50 cursor-pointer"
                >
                  <td className="py-2 font-mono text-xs">{e.explanation_id.slice(0, 8)}...</td>
                  <td className="py-2">
                    <span
                      className={`px-2 py-0.5 rounded text-xs font-medium ${methodColors[e.method] || "bg-gray-100 text-gray-700"}`}
                    >
                      {e.method.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-2 font-mono text-xs text-gray-500">
                    {e.inference_event_id.slice(0, 8)}...
                  </td>
                  <td className="py-2 text-gray-500">{new Date(e.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Detail view */}
      {selectedDetail && (
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold">Explanation Detail</h3>
            <button
              onClick={() => setSelectedDetail(null)}
              className="text-gray-400 hover:text-gray-600 text-sm"
            >
              Close
            </button>
          </div>
          <pre className="text-xs bg-gray-50 rounded p-2 overflow-x-auto">
            {JSON.stringify(selectedDetail, null, 2)}
          </pre>
        </div>
      )}

      {/* Trigger LIME */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Generate LIME Explanation</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm">
            Event ID
            <input
              type="text"
              value={limeEventId}
              onChange={(e) => setLimeEventId(e.target.value)}
              className="border rounded px-3 py-1 w-72 block mt-1 font-mono text-xs"
              placeholder="Inference event UUID"
            />
          </label>
          <label className="text-sm">
            Superpixels
            <input
              type="number"
              value={limeSuperpixels}
              onChange={(e) => setLimeSuperpixels(+e.target.value)}
              className="border rounded px-2 py-1 w-20 block mt-1"
              min={4}
            />
          </label>
          <label className="text-sm">
            Samples
            <input
              type="number"
              value={limeSamples}
              onChange={(e) => setLimeSamples(+e.target.value)}
              className="border rounded px-2 py-1 w-20 block mt-1"
              min={10}
            />
          </label>
          <label className="text-sm">
            Top K
            <input
              type="number"
              value={limeTopK}
              onChange={(e) => setLimeTopK(+e.target.value)}
              className="border rounded px-2 py-1 w-20 block mt-1"
              min={1}
            />
          </label>
        </div>
        <button onClick={triggerLime} className="px-4 py-1 bg-green-600 text-white rounded text-sm">
          Run LIME
        </button>
        {limeResult && (
          <pre className="text-xs bg-gray-50 rounded p-2 overflow-x-auto">
            {JSON.stringify(limeResult, null, 2)}
          </pre>
        )}
      </div>

      {/* Trigger SHAP */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Generate SHAP Explanation</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm">
            Event ID
            <input
              type="text"
              value={shapEventId}
              onChange={(e) => setShapEventId(e.target.value)}
              className="border rounded px-3 py-1 w-72 block mt-1 font-mono text-xs"
              placeholder="Inference event UUID"
            />
          </label>
          <label className="text-sm">
            Samples
            <input
              type="number"
              value={shapSamples}
              onChange={(e) => setShapSamples(+e.target.value)}
              className="border rounded px-2 py-1 w-20 block mt-1"
              min={10}
            />
          </label>
          <label className="text-sm">
            Random State
            <input
              type="number"
              value={shapRandomState}
              onChange={(e) => setShapRandomState(+e.target.value)}
              className="border rounded px-2 py-1 w-20 block mt-1"
            />
          </label>
        </div>
        <button onClick={triggerShap} className="px-4 py-1 bg-purple-600 text-white rounded text-sm">
          Run SHAP
        </button>
        {shapResult && (
          <pre className="text-xs bg-gray-50 rounded p-2 overflow-x-auto">
            {JSON.stringify(shapResult, null, 2)}
          </pre>
        )}
      </div>

      {/* Compare explanations */}
      <div className="bg-white rounded-lg shadow p-4 space-y-2">
        <h3 className="font-semibold">Compare Explanations</h3>
        <p className="text-xs text-gray-500">
          Compare LIME, SHAP, and SDD explanations side-by-side for the same inference event.
        </p>
        <div className="flex gap-4 items-end">
          <label className="text-sm">
            Inference Event ID
            <input
              type="text"
              value={compareEventId}
              onChange={(e) => setCompareEventId(e.target.value)}
              className="border rounded px-3 py-1 w-72 block mt-1 font-mono text-xs"
              placeholder="Inference event UUID"
            />
          </label>
          <button
            onClick={loadComparison}
            className="px-4 py-1 bg-blue-600 text-white rounded text-sm"
          >
            Compare
          </button>
        </div>
      </div>

      {/* Comparison results */}
      {comparison && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold mb-3">
            Comparison for{" "}
            <span className="font-mono text-xs">{comparison.inference_event_id.slice(0, 12)}...</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <CompareColumn
              title="LIME"
              count={comparison.total_lime}
              items={comparison.lime_explanations}
              color="green"
            />
            <CompareColumn
              title="SHAP"
              count={comparison.total_shap}
              items={comparison.shap_explanations}
              color="purple"
            />
            <CompareColumn
              title="SDD"
              count={comparison.total_sdd}
              items={comparison.sdd_explanations}
              color="blue"
            />
          </div>
        </div>
      )}
    </div>
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
  const borderColor = `border-${color}-300`;
  const bgColor = `bg-${color}-50`;
  const textColor = `text-${color}-700`;

  return (
    <div className={`rounded-lg border-2 ${borderColor} ${bgColor} p-3`}>
      <div className="flex items-center justify-between mb-2">
        <span className={`font-semibold ${textColor}`}>{title}</span>
        <span className="text-xs text-gray-500">{count} result{count !== 1 ? "s" : ""}</span>
      </div>
      {items.length === 0 ? (
        <p className="text-xs text-gray-400">No explanations</p>
      ) : (
        <div className="space-y-2">
          {items.map((item) => (
            <div key={item.explanation_id} className="bg-white rounded p-2 text-xs">
              <p className="font-mono text-gray-600">{item.explanation_id.slice(0, 12)}...</p>
              <p className="text-gray-400">{new Date(item.timestamp).toLocaleString()}</p>
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
