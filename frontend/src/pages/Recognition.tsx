import { useEffect, useState } from "react";

interface MatchResult {
  subject_id: string;
  distance: number;
  similarity: number;
}

interface PredictResult {
  predicted_label: number;
  confidence: number;
  model_id: string;
  matches?: MatchResult[];
  gallery_matches?: number;
}

interface TrainResult {
  model_id: string;
  accuracy: number;
  num_classes: number;
  training_time_ms: number;
}

interface GallerySubject {
  subject_id: string;
  entries: number;
  enrolled_at: string;
}

interface RecognitionModel {
  model_id: string;
  model_type: string;
  accuracy: number;
  trained_at: string;
  num_classes: number;
}

interface ModelDetail {
  model_id: string;
  model_type: string;
  accuracy: number;
  trained_at: string;
  num_classes: number;
  provenance_chain?: Array<Record<string, unknown>>;
}

type TabId = "predict" | "train" | "gallery" | "models";

export default function Recognition() {
  const [activeTab, setActiveTab] = useState<TabId>("predict");

  const tabs: { id: TabId; label: string }[] = [
    { id: "predict", label: "Predict" },
    { id: "train", label: "Train" },
    { id: "gallery", label: "Gallery" },
    { id: "models", label: "Models" },
  ];

  return (
    <div className="space-y-6">
      {/* Page title shown in header bar */}

      {/* Tab bar */}
      <div className="flex gap-1 bg-surface rounded-lg p-1">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === t.id
                ? "bg-cyan/10 text-cyan border border-cyan/20"
                : "text-fg-muted hover:text-fg hover:bg-fg/5 border border-transparent"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {activeTab === "predict" && <PredictTab />}
      {activeTab === "train" && <TrainTab />}
      {activeTab === "gallery" && <GalleryTab />}
      {activeTab === "models" && <ModelsTab />}
    </div>
  );
}

function PredictTab() {
  const [file, setFile] = useState<File | null>(null);
  const [topK, setTopK] = useState(5);
  const [demographicGroup, setDemographicGroup] = useState("");
  const [trueLabel, setTrueLabel] = useState("");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handlePredict = () => {
    if (!file) {
      setError("Please select an image file");
      return;
    }
    setError("");
    setResult(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("image", file);

    const params = new URLSearchParams();
    params.set("top_k", String(topK));
    if (demographicGroup.trim()) params.set("demographic_group", demographicGroup.trim());
    if (trueLabel.trim()) params.set("true_label", trueLabel.trim());

    fetch(`/api/v1/recognition/predict?${params.toString()}`, {
      method: "POST",
      body: formData,
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
        setError(`Prediction error: ${e.message}`);
        setLoading(false);
      });
  };

  return (
    <div className="space-y-4">
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Predict Face</h3>
        <p className="text-xs text-fg-muted">Upload an image to identify a face using the trained model.</p>

        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Image
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="ff-input block mt-1"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Top K
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(+e.target.value)}
              className="ff-input w-20 block mt-1"
              min={1}
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Demographic Group
            <input
              type="text"
              value={demographicGroup}
              onChange={(e) => setDemographicGroup(e.target.value)}
              className="ff-input w-40 block mt-1"
              placeholder="Optional"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            True Label
            <input
              type="text"
              value={trueLabel}
              onChange={(e) => setTrueLabel(e.target.value)}
              className="ff-input w-24 block mt-1"
              placeholder="Optional"
            />
          </label>
        </div>

        <button onClick={handlePredict} disabled={loading || !file} className="btn-primary">
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="glass-card p-4 space-y-3">
          <h3 className="font-semibold text-fg-secondary">Prediction Result</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-fg-muted">Predicted Label</p>
              <p className="text-lg font-bold text-cyan">{result.predicted_label}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Confidence</p>
              <p className={`text-lg font-bold ${result.confidence >= 0.8 ? "text-teal" : result.confidence >= 0.5 ? "text-gold" : "text-rose-ember"}`}>
                {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Model</p>
              <p className="text-sm font-mono text-fg-secondary">{result.model_id?.slice(0, 12) || "—"}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Gallery Matches</p>
              <p className="text-lg font-bold text-amethyst">{result.gallery_matches ?? 0}</p>
            </div>
          </div>

          {result.matches && result.matches.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-fg-muted mb-2">Matches</h4>
              <div className="space-y-1">
                {result.matches.map((m, i) => (
                  <div key={i} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                    <span className="text-sm text-fg-secondary font-mono">{m.subject_id}</span>
                    <div className="text-right">
                      <span className={`text-sm font-semibold ${m.similarity >= 0.8 ? "text-teal" : m.similarity >= 0.5 ? "text-gold" : "text-rose-ember"}`}>
                        {(m.similarity * 100).toFixed(1)}%
                      </span>
                      <span className="text-xs text-fg-faint ml-2">d={m.distance.toFixed(4)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TrainTab() {
  const [datasetPath, setDatasetPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [nComponents, setNComponents] = useState(50);
  const [labels, setLabels] = useState("");
  const [result, setResult] = useState<TrainResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleTrain = () => {
    if (!datasetPath.trim()) {
      setError("Dataset path is required");
      return;
    }
    setError("");
    setResult(null);
    setLoading(true);

    const body: Record<string, unknown> = {
      dataset_path: datasetPath.trim(),
      n_components: nComponents,
    };
    if (outputDir.trim()) body.output_dir = outputDir.trim();
    if (labels.trim()) {
      try {
        body.labels = JSON.parse(labels);
      } catch {
        setError("Invalid JSON for labels");
        setLoading(false);
        return;
      }
    }

    fetch("/api/v1/recognition/train", {
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
        setError(`Training error: ${e.message}`);
        setLoading(false);
      });
  };

  return (
    <div className="space-y-4">
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Train Model</h3>
        <p className="text-xs text-fg-muted">Train a PCA+SVM face recognition model on a local dataset.</p>

        <div className="space-y-3">
          <label className="text-sm text-fg-secondary block">
            Dataset Path
            <input
              type="text"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              className="ff-input w-full block mt-1"
              placeholder="/path/to/dataset"
            />
          </label>
          <label className="text-sm text-fg-secondary block">
            Output Directory
            <input
              type="text"
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              className="ff-input w-full block mt-1"
              placeholder="Optional — /path/to/output"
            />
          </label>
          <div className="flex gap-4">
            <label className="text-sm text-fg-secondary">
              PCA Components
              <input
                type="number"
                value={nComponents}
                onChange={(e) => setNComponents(+e.target.value)}
                className="ff-input w-24 block mt-1"
                min={1}
              />
            </label>
          </div>
          <label className="text-sm text-fg-secondary block">
            Labels (JSON array)
            <input
              type="text"
              value={labels}
              onChange={(e) => setLabels(e.target.value)}
              className="ff-input w-full font-mono text-xs block mt-1"
              placeholder='Optional — [0, 1, 0, 2, ...]'
            />
          </label>
        </div>

        <button onClick={handleTrain} disabled={loading} className="btn-primary">
          {loading ? "Training..." : "Train Model"}
        </button>
      </div>

      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="glass-card p-4">
          <h3 className="font-semibold text-fg-secondary mb-3">Training Result</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-fg-muted">Model ID</p>
              <p className="text-sm font-mono text-fg-secondary">{result.model_id.slice(0, 12)}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Accuracy</p>
              <p className={`text-lg font-bold ${result.accuracy >= 0.9 ? "text-teal" : result.accuracy >= 0.7 ? "text-gold" : "text-rose-ember"}`}>
                {(result.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Classes</p>
              <p className="text-lg font-bold text-amethyst">{result.num_classes}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Time</p>
              <p className="text-lg font-bold text-gold">{result.training_time_ms}ms</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function GalleryTab() {
  const [subjects, setSubjects] = useState<GallerySubject[]>([]);
  const [galleryCount, setGalleryCount] = useState<number | null>(null);
  const [error, setError] = useState("");

  // Enroll state
  const [enrollFile, setEnrollFile] = useState<File | null>(null);
  const [enrollSubjectId, setEnrollSubjectId] = useState("");
  const [enrollResult, setEnrollResult] = useState<Record<string, unknown> | null>(null);
  const [enrolling, setEnrolling] = useState(false);

  // Search state
  const [searchFile, setSearchFile] = useState<File | null>(null);
  const [searchTopK, setSearchTopK] = useState(5);
  const [searchResults, setSearchResults] = useState<MatchResult[] | null>(null);
  const [searching, setSearching] = useState(false);

  const fetchSubjects = () => {
    fetch("/api/v1/gallery/subjects")
      .then((r) => r.json())
      .then((data) => setSubjects(Array.isArray(data) ? data : data.subjects || []))
      .catch(() => {});
  };

  const fetchCount = () => {
    fetch("/api/v1/gallery/count")
      .then((r) => r.json())
      .then((data) => setGalleryCount(data.total ?? data.count ?? 0))
      .catch(() => {});
  };

  useEffect(() => {
    fetchSubjects();
    fetchCount();
  }, []);

  const handleEnroll = () => {
    if (!enrollFile || !enrollSubjectId.trim()) {
      setError("Image and subject ID are required for enrollment");
      return;
    }
    setError("");
    setEnrollResult(null);
    setEnrolling(true);

    const formData = new FormData();
    formData.append("image", enrollFile);
    formData.append("subject_id", enrollSubjectId.trim());

    fetch("/api/v1/gallery/enroll", {
      method: "POST",
      body: formData,
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setEnrollResult(data);
        setEnrolling(false);
        fetchSubjects();
        fetchCount();
      })
      .catch((e) => {
        setError(`Enroll error: ${e.message}`);
        setEnrolling(false);
      });
  };

  const handleSearch = () => {
    if (!searchFile) {
      setError("Please select an image to search");
      return;
    }
    setError("");
    setSearchResults(null);
    setSearching(true);

    const formData = new FormData();
    formData.append("image", searchFile);

    fetch(`/api/v1/gallery/search?top_k=${searchTopK}`, {
      method: "POST",
      body: formData,
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setSearchResults(data.matches || data.results || []);
        setSearching(false);
      })
      .catch((e) => {
        setError(`Search error: ${e.message}`);
        setSearching(false);
      });
  };

  const deleteSubject = (subjectId: string) => {
    fetch(`/api/v1/gallery/subjects/${subjectId}`, { method: "DELETE" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        fetchSubjects();
        fetchCount();
      })
      .catch((e) => setError(`Delete error: ${e.message}`));
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Gallery stats */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-fg-secondary">Gallery</h3>
          <span className="text-sm text-fg-muted">
            {galleryCount !== null ? `${galleryCount} entries` : "Loading..."}
          </span>
        </div>
      </div>

      {/* Enroll */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Enroll Face</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Image
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setEnrollFile(e.target.files?.[0] || null)}
              className="ff-input block mt-1"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Subject ID
            <input
              type="text"
              value={enrollSubjectId}
              onChange={(e) => setEnrollSubjectId(e.target.value)}
              className="ff-input w-48 block mt-1"
              placeholder="e.g. person_001"
            />
          </label>
        </div>
        <button onClick={handleEnroll} disabled={enrolling} className="btn-success">
          {enrolling ? "Enrolling..." : "Enroll"}
        </button>
        {enrollResult && (
          <pre className="text-xs bg-surface rounded-lg p-2 overflow-x-auto">
            {JSON.stringify(enrollResult, null, 2)}
          </pre>
        )}
      </div>

      {/* Search */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Search Gallery</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Image
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setSearchFile(e.target.files?.[0] || null)}
              className="ff-input block mt-1"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Top K
            <input
              type="number"
              value={searchTopK}
              onChange={(e) => setSearchTopK(+e.target.value)}
              className="ff-input w-20 block mt-1"
              min={1}
            />
          </label>
        </div>
        <button onClick={handleSearch} disabled={searching} className="btn-primary">
          {searching ? "Searching..." : "Search"}
        </button>
        {searchResults && (
          <div className="mt-2 space-y-1">
            {searchResults.length === 0 ? (
              <p className="text-sm text-fg-faint">No matches found</p>
            ) : (
              searchResults.map((m, i) => (
                <div key={i} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                  <span className="text-sm text-fg-secondary font-mono">{m.subject_id}</span>
                  <div className="text-right">
                    <span className={`text-sm font-semibold ${m.similarity >= 0.8 ? "text-teal" : m.similarity >= 0.5 ? "text-gold" : "text-rose-ember"}`}>
                      {(m.similarity * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs text-fg-faint ml-2">d={m.distance.toFixed(4)}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Subjects list */}
      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Enrolled Subjects</h3>
        {subjects.length === 0 ? (
          <p className="text-sm text-fg-faint">No subjects enrolled yet</p>
        ) : (
          <div className="space-y-2">
            {subjects.map((s) => (
              <div key={s.subject_id} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                <div>
                  <p className="text-sm text-fg-secondary font-mono">{s.subject_id}</p>
                  <p className="text-xs text-fg-faint">
                    {s.entries} entr{s.entries !== 1 ? "ies" : "y"}
                    {s.enrolled_at && ` — ${new Date(s.enrolled_at).toLocaleDateString()}`}
                  </p>
                </div>
                <button
                  onClick={() => deleteSubject(s.subject_id)}
                  className="text-xs text-rose-ember hover:text-rose-ember/80 px-2 py-1 rounded hover:bg-rose-ember/10 transition-colors"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ModelsTab() {
  const [models, setModels] = useState<RecognitionModel[]>([]);
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/v1/recognition/models")
      .then((r) => r.json())
      .then((data) => setModels(Array.isArray(data) ? data : []))
      .catch((e) => setError(e.message));
  }, []);

  const viewDetail = (modelId: string) => {
    fetch(`/api/v1/recognition/models/${modelId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setDetail)
      .catch((e) => setError(e.message));
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Model Registry</h3>
        {models.length === 0 ? (
          <p className="text-sm text-fg-faint">No models trained yet</p>
        ) : (
          <div className="space-y-2">
            {models.map((m) => (
              <div
                key={m.model_id}
                onClick={() => viewDetail(m.model_id)}
                className="flex items-center justify-between bg-surface rounded-lg px-3 py-2 hover:bg-fg/[0.02] cursor-pointer"
              >
                <div>
                  <p className="text-sm text-fg-secondary font-mono">{m.model_id.slice(0, 16)}</p>
                  <p className="text-xs text-fg-faint">
                    {m.model_type} — {m.num_classes} classes
                  </p>
                </div>
                <div className="text-right">
                  <p className={`text-sm font-semibold ${m.accuracy >= 0.9 ? "text-teal" : m.accuracy >= 0.7 ? "text-gold" : "text-rose-ember"}`}>
                    {(m.accuracy * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-fg-faint">{new Date(m.trained_at).toLocaleDateString()}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {detail && (
        <div className="glass-card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-fg-secondary">Model Detail</h3>
            <button onClick={() => setDetail(null)} className="text-fg-faint hover:text-fg-secondary text-sm">
              Close
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div>
              <p className="text-xs text-fg-muted">Model ID</p>
              <p className="text-sm font-mono text-fg-secondary">{detail.model_id.slice(0, 16)}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Type</p>
              <p className="text-sm text-fg-secondary">{detail.model_type}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Accuracy</p>
              <p className={`text-lg font-bold ${detail.accuracy >= 0.9 ? "text-teal" : detail.accuracy >= 0.7 ? "text-gold" : "text-rose-ember"}`}>
                {(detail.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Trained</p>
              <p className="text-sm text-fg-secondary">{new Date(detail.trained_at).toLocaleString()}</p>
            </div>
          </div>

          {detail.provenance_chain && detail.provenance_chain.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-fg-muted mb-2">Provenance Chain</h4>
              <div className="space-y-2">
                {detail.provenance_chain.map((node, i) => (
                  <div key={i} className="bg-surface rounded-lg p-2 text-xs">
                    <pre className="overflow-x-auto">{JSON.stringify(node, null, 2)}</pre>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
