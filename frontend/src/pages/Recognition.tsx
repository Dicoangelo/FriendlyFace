import { useEffect, useState } from "react";
import EmptyState from "../components/EmptyState";

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

interface LivenessResult {
  is_live: boolean;
  score: number;
  checks: Record<string, number>;
  details: string;
  threshold: number;
}

type TabId = "predict" | "train" | "gallery" | "models" | "voice" | "liveness";

export default function Recognition() {
  const [activeTab, setActiveTab] = useState<TabId>("predict");

  const tabs: { id: TabId; label: string }[] = [
    { id: "predict", label: "Predict" },
    { id: "train", label: "Train" },
    { id: "gallery", label: "Gallery" },
    { id: "models", label: "Models" },
    { id: "voice", label: "Voice" },
    { id: "liveness", label: "Liveness" },
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

      <div key={activeTab} className="animate-fade-in">
        {activeTab === "predict" && <PredictTab />}
        {activeTab === "train" && <TrainTab />}
        {activeTab === "gallery" && <GalleryTab />}
        {activeTab === "models" && <ModelsTab />}
        {activeTab === "voice" && <VoiceTab />}
        {activeTab === "liveness" && <LivenessTab />}
      </div>
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
        <div className="glass-card p-4 space-y-3 animate-fade-in">
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
        <div className="glass-card p-4 animate-fade-in">
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
          <div className="flex items-center gap-2 text-teal text-sm bg-teal/10 rounded-lg px-3 py-2 animate-fade-in">
            <span className="font-bold">&#10003;</span>
            <span>Voice enrolled — {String(enrollResult.subject_id || "success")}</span>
          </div>
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
          <EmptyState title="No subjects enrolled yet" subtitle="Enroll voice biometrics via the API" />
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

function VoiceTab() {
  const [error, setError] = useState("");

  // Enroll state
  const [enrollFile, setEnrollFile] = useState<File | null>(null);
  const [enrollSubjectId, setEnrollSubjectId] = useState("");
  const [enrollResult, setEnrollResult] = useState<Record<string, unknown> | null>(null);
  const [enrolling, setEnrolling] = useState(false);

  // Verify state
  const [verifyFile, setVerifyFile] = useState<File | null>(null);
  const [verifyTopK, setVerifyTopK] = useState(3);
  const [verifyResult, setVerifyResult] = useState<Record<string, unknown> | null>(null);
  const [verifying, setVerifying] = useState(false);

  // Fusion state
  const [faceLabel, setFaceLabel] = useState("");
  const [faceConf, setFaceConf] = useState(0.8);
  const [voiceConf, setVoiceConf] = useState(0.7);
  const [faceWeight, setFaceWeight] = useState(0.6);
  const [fusionResult, setFusionResult] = useState<Record<string, unknown> | null>(null);
  const [fusing, setFusing] = useState(false);

  const handleEnroll = () => {
    if (!enrollFile || !enrollSubjectId.trim()) {
      setError("Audio file and subject ID are required");
      return;
    }
    setError("");
    setEnrollResult(null);
    setEnrolling(true);

    const formData = new FormData();
    formData.append("audio", enrollFile);

    fetch(`/api/v1/recognition/voice/enroll?subject_id=${encodeURIComponent(enrollSubjectId.trim())}`, {
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
      })
      .catch((e) => {
        setError(`Enroll error: ${e.message}`);
        setEnrolling(false);
      });
  };

  const handleVerify = () => {
    if (!verifyFile) {
      setError("Please select an audio file to verify");
      return;
    }
    setError("");
    setVerifyResult(null);
    setVerifying(true);

    const formData = new FormData();
    formData.append("audio", verifyFile);

    fetch(`/api/v1/recognition/voice/verify?top_k=${verifyTopK}`, {
      method: "POST",
      body: formData,
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setVerifyResult(data);
        setVerifying(false);
      })
      .catch((e) => {
        setError(`Verify error: ${e.message}`);
        setVerifying(false);
      });
  };

  const handleFusion = () => {
    if (!faceLabel.trim()) {
      setError("Face label is required for fusion");
      return;
    }
    setError("");
    setFusionResult(null);
    setFusing(true);

    fetch("/api/v1/recognition/multimodal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        face_label: faceLabel.trim(),
        face_confidence: faceConf,
        voice_confidence: voiceConf,
        face_weight: faceWeight,
        voice_weight: +(1 - faceWeight).toFixed(2),
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setFusionResult(data);
        setFusing(false);
      })
      .catch((e) => {
        setError(`Fusion error: ${e.message}`);
        setFusing(false);
      });
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Voice Enroll */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Enroll Voice</h3>
        <p className="text-xs text-fg-muted">Upload PCM/WAV audio to enroll a voice biometric for a subject.</p>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Audio File
            <input
              type="file"
              accept="audio/*,.wav,.pcm,.raw"
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
              placeholder="e.g. speaker_001"
            />
          </label>
        </div>
        <button onClick={handleEnroll} disabled={enrolling || !enrollFile} className="btn-success">
          {enrolling ? "Enrolling..." : "Enroll Voice"}
        </button>
        {enrollResult && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-2">
            <div>
              <p className="text-xs text-fg-muted">Subject</p>
              <p className="text-sm font-mono text-fg-secondary">{String(enrollResult.subject_id)}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Embedding Dim</p>
              <p className="text-lg font-bold text-cyan">{String(enrollResult.embedding_dim)}</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Duration</p>
              <p className="text-lg font-bold text-gold">{Number(enrollResult.duration_seconds).toFixed(2)}s</p>
            </div>
            <div>
              <p className="text-xs text-fg-muted">Total Enrolled</p>
              <p className="text-lg font-bold text-teal">{String(enrollResult.total_enrolled)}</p>
            </div>
          </div>
        )}
      </div>

      {/* Voice Verify */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Verify Voice</h3>
        <p className="text-xs text-fg-muted">Upload audio to identify against enrolled voices using MFCC embeddings.</p>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Audio File
            <input
              type="file"
              accept="audio/*,.wav,.pcm,.raw"
              onChange={(e) => setVerifyFile(e.target.files?.[0] || null)}
              className="ff-input block mt-1"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Top K
            <input
              type="number"
              value={verifyTopK}
              onChange={(e) => setVerifyTopK(+e.target.value)}
              className="ff-input w-20 block mt-1"
              min={1}
            />
          </label>
        </div>
        <button onClick={handleVerify} disabled={verifying || !verifyFile} className="btn-primary">
          {verifying ? "Verifying..." : "Verify"}
        </button>
        {verifyResult && (
          <div className="mt-2">
            <p className="text-xs text-fg-muted mb-2">Input hash: <span className="font-mono">{String(verifyResult.input_hash).slice(0, 16)}...</span></p>
            {Array.isArray(verifyResult.matches) && verifyResult.matches.length > 0 ? (
              <div className="space-y-1">
                {(verifyResult.matches as Array<{ label: string; confidence: number }>).map((m, i) => (
                  <div key={i} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                    <span className="text-sm text-fg-secondary font-mono">{m.label}</span>
                    <span className={`text-sm font-semibold ${m.confidence >= 0.8 ? "text-teal" : m.confidence >= 0.5 ? "text-gold" : "text-rose-ember"}`}>
                      {(m.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-fg-faint">No matches found</p>
            )}
          </div>
        )}
      </div>

      {/* Multi-modal Fusion */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Multi-Modal Fusion</h3>
        <p className="text-xs text-fg-muted">Fuse face and voice confidence scores into a unified decision.</p>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Face Label
            <input
              type="text"
              value={faceLabel}
              onChange={(e) => setFaceLabel(e.target.value)}
              className="ff-input w-40 block mt-1"
              placeholder="e.g. person_001"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Face Confidence
            <input
              type="number"
              value={faceConf}
              onChange={(e) => setFaceConf(+e.target.value)}
              className="ff-input w-24 block mt-1"
              min={0} max={1} step={0.05}
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Voice Confidence
            <input
              type="number"
              value={voiceConf}
              onChange={(e) => setVoiceConf(+e.target.value)}
              className="ff-input w-24 block mt-1"
              min={0} max={1} step={0.05}
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Face Weight
            <input
              type="number"
              value={faceWeight}
              onChange={(e) => setFaceWeight(+e.target.value)}
              className="ff-input w-24 block mt-1"
              min={0.01} max={0.99} step={0.05}
            />
          </label>
        </div>
        <button onClick={handleFusion} disabled={fusing} className="btn-accent">
          {fusing ? "Fusing..." : "Run Fusion"}
        </button>
        {fusionResult && (
          <div className="mt-2">
            <p className="text-xs text-fg-muted mb-2">
              Method: <span className="text-fg-secondary">{String(fusionResult.fusion_method)}</span>
              {" | "}Face weight: <span className="text-fg-secondary">{String(fusionResult.face_weight)}</span>
              {" | "}Voice weight: <span className="text-fg-secondary">{String(fusionResult.voice_weight)}</span>
            </p>
            {Array.isArray(fusionResult.fused_matches) && (
              <div className="space-y-1">
                {(fusionResult.fused_matches as Array<{ label: string; fused_confidence: number; face_confidence: number; voice_confidence: number }>).map((m, i) => (
                  <div key={i} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                    <span className="text-sm text-fg-secondary font-mono">{m.label}</span>
                    <div className="flex items-center gap-4 text-xs">
                      <span className="text-fg-muted">Face: {(m.face_confidence * 100).toFixed(0)}%</span>
                      <span className="text-fg-muted">Voice: {(m.voice_confidence * 100).toFixed(0)}%</span>
                      <span className={`text-sm font-semibold ${m.fused_confidence >= 0.8 ? "text-teal" : m.fused_confidence >= 0.5 ? "text-gold" : "text-rose-ember"}`}>
                        Fused: {(m.fused_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function LivenessTab() {
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState<LivenessResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleCheck = () => {
    if (!file) {
      setError("Please select a face image");
      return;
    }
    setError("");
    setResult(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("image", file);

    fetch(`/api/v1/recognition/liveness?threshold=${threshold}`, {
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
        setError(`Liveness check error: ${e.message}`);
        setLoading(false);
      });
  };

  return (
    <div className="space-y-4">
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Passive Liveness Detection</h3>
        <p className="text-xs text-fg-muted">
          Anti-spoofing analysis: moire pattern, frequency spectrum, and color distribution checks.
        </p>

        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Face Image
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="ff-input block mt-1"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Threshold
            <input
              type="number"
              value={threshold}
              onChange={(e) => setThreshold(+e.target.value)}
              className="ff-input w-24 block mt-1"
              min={0}
              max={1}
              step={0.05}
            />
          </label>
        </div>

        <button onClick={handleCheck} disabled={loading || !file} className="btn-primary">
          {loading ? "Checking..." : "Check Liveness"}
        </button>
      </div>

      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="glass-card p-4 space-y-3 animate-fade-in">
          <div className="flex items-center gap-3">
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold ${
                result.is_live
                  ? "bg-teal/20 text-teal"
                  : "bg-rose-ember/20 text-rose-ember"
              }`}
            >
              {result.is_live ? "\u2713" : "\u2717"}
            </div>
            <div>
              <p className={`text-lg font-bold ${result.is_live ? "text-teal" : "text-rose-ember"}`}>
                {result.is_live ? "Live" : "Spoof Detected"}
              </p>
              <p className="text-xs text-fg-muted">
                Score: {result.score.toFixed(3)} | Threshold: {result.threshold}
              </p>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-fg-muted mb-2">Check Scores</h4>
            <div className="space-y-2">
              {Object.entries(result.checks).map(([name, score]) => (
                <div key={name} className="flex items-center gap-3">
                  <span className="text-xs text-fg-secondary w-28 truncate">{name}</span>
                  <div className="flex-1 bg-surface rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        score >= 0.5 ? "bg-teal" : "bg-rose-ember"
                      }`}
                      style={{ width: `${Math.min(score * 100, 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-fg-muted w-12 text-right">
                    {(score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {result.details && (
            <p className="text-xs text-fg-faint">{result.details}</p>
          )}
        </div>
      )}
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
          <EmptyState title="No models trained yet" subtitle="Train a PCA+SVM model from the Train tab" />
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
