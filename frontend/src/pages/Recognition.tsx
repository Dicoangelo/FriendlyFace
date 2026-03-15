import { useEffect, useState } from "react";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import ConfirmDialog from "../components/ConfirmDialog";
import { SkeletonRow } from "../components/Skeleton";

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
  predictions?: Array<{ label: string; confidence: number }>;
}

interface TrainResult {
  model_id: string;
  accuracy: number;
  num_classes: number;
  training_time_ms: number;
  event_id?: string;
}

interface GallerySubject {
  subject_id: string;
  entries: number;
  image_count?: number;
  enrolled_at: string;
}

interface RecognitionModel {
  model_id: string;
  model_type: string;
  accuracy: number;
  trained_at: string;
  num_classes: number;
}

type TabId = "gallery" | "train" | "recognize" | "models" | "voice" | "liveness";

export default function Recognition() {
  const [activeTab, setActiveTab] = useState<TabId>("gallery");

  const tabs: { id: TabId; label: string }[] = [
    { id: "gallery", label: "Gallery" },
    { id: "train", label: "Train" },
    { id: "recognize", label: "Recognize" },
    { id: "models", label: "Models" },
    { id: "voice", label: "Voice" },
    { id: "liveness", label: "Liveness" },
  ];

  return (
    <div className="space-y-6">
      {/* Pill tab bar */}
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
        {activeTab === "gallery" && <GalleryTab />}
        {activeTab === "train" && <TrainTab />}
        {activeTab === "recognize" && <RecognizeTab />}
        {activeTab === "models" && <ModelsTab />}
        {activeTab === "voice" && <VoiceTab />}
        {activeTab === "liveness" && <LivenessTab />}
      </div>
    </div>
  );
}

/* ── Gallery Tab ── */
function GalleryTab() {
  const [subjects, setSubjects] = useState<GallerySubject[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const fetchSubjects = () => {
    setLoading(true);
    fetch("/api/v1/gallery/subjects")
      .then((r) => r.json())
      .then((data) => {
        setSubjects(Array.isArray(data) ? data : data.subjects || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  useEffect(() => { fetchSubjects(); }, []);

  const deleteSubject = (subjectId: string) => {
    fetch(`/api/v1/gallery/subjects/${subjectId}`, { method: "DELETE" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        setDeleteTarget(null);
        fetchSubjects();
      })
      .catch((e) => {
        setError(`Delete error: ${e.message}`);
        setDeleteTarget(null);
      });
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">{error}</div>
      )}

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Delete Subject"
        message={`Are you sure you want to delete subject "${deleteTarget}"? This cannot be undone.`}
        variant="danger"
        confirmLabel="Delete"
        onConfirm={() => deleteTarget && deleteSubject(deleteTarget)}
        onCancel={() => setDeleteTarget(null)}
      />

      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Enrolled Subjects</h3>
        {loading ? (
          <div className="space-y-1">{[...Array(3)].map((_, i) => <SkeletonRow key={i} />)}</div>
        ) : subjects.length === 0 ? (
          <EmptyState title="No subjects enrolled" subtitle="Enroll faces via the API or Gallery search" />
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {subjects.map((s) => (
              <div key={s.subject_id} className="bg-surface rounded-lg p-3 flex items-center justify-between">
                <div>
                  <p className="text-sm text-fg-secondary font-mono">{s.subject_id}</p>
                  <p className="text-xs text-fg-faint">
                    {s.image_count ?? s.entries} image{(s.image_count ?? s.entries) !== 1 ? "s" : ""}
                    {s.enrolled_at && ` — ${new Date(s.enrolled_at).toLocaleDateString()}`}
                  </p>
                </div>
                <button
                  onClick={() => setDeleteTarget(s.subject_id)}
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

/* ── Train Tab ── */
function TrainTab() {
  const [classes, setClasses] = useState<Array<{ name: string; count: number }>>([{ name: "", count: 10 }]);
  const [epochs, setEpochs] = useState(10);
  const [nComponents, setNComponents] = useState(50);
  const [result, setResult] = useState<TrainResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const addRow = () => setClasses([...classes, { name: "", count: 10 }]);
  const removeRow = (i: number) => setClasses(classes.filter((_, idx) => idx !== i));
  const updateRow = (i: number, field: "name" | "count", value: string | number) => {
    const updated = [...classes];
    if (field === "name") updated[i].name = value as string;
    else updated[i].count = value as number;
    setClasses(updated);
  };

  const handleTrain = () => {
    const validClasses = classes.filter((c) => c.name.trim());
    if (validClasses.length === 0) {
      setError("At least one class is required");
      return;
    }
    setError("");
    setResult(null);
    setLoading(true);

    fetch("/api/v1/recognition/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classes: validClasses.map((c) => ({ class_name: c.name.trim(), image_count: c.count })),
        epochs,
        n_components: nComponents,
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => { setResult(data); setLoading(false); })
      .catch((e) => { setError(`Training error: ${e.message}`); setLoading(false); });
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">{error}</div>
      )}

      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Train Model</h3>
        <p className="text-xs text-fg-muted">Define classes and train a PCA+SVM face recognition model.</p>

        {/* Class builder */}
        <div className="space-y-2">
          <p className="text-sm text-fg-secondary font-medium">Classes</p>
          {classes.map((cls, i) => (
            <div key={i} className="flex gap-2 items-end">
              <label className="text-xs text-fg-muted flex-1">
                {i === 0 && "Class Name"}
                <input
                  type="text"
                  value={cls.name}
                  onChange={(e) => updateRow(i, "name", e.target.value)}
                  className="ff-input w-full block mt-0.5"
                  placeholder={`class_${i}`}
                />
              </label>
              <label className="text-xs text-fg-muted w-24">
                {i === 0 && "Images"}
                <input
                  type="number"
                  value={cls.count}
                  onChange={(e) => updateRow(i, "count", +e.target.value)}
                  className="ff-input w-full block mt-0.5"
                  min={1}
                />
              </label>
              {classes.length > 1 && (
                <button
                  onClick={() => removeRow(i)}
                  className="text-xs text-rose-ember hover:text-rose-ember/80 px-2 py-2 rounded hover:bg-rose-ember/10 transition-colors"
                >
                  Remove
                </button>
              )}
            </div>
          ))}
          <button onClick={addRow} className="text-xs text-cyan hover:text-cyan/80">+ Add class</button>
        </div>

        <div className="flex gap-4">
          <label className="text-sm text-fg-secondary">
            Epochs
            <input type="number" value={epochs} onChange={(e) => setEpochs(+e.target.value)} className="ff-input w-20 block mt-1" min={1} />
          </label>
          <label className="text-sm text-fg-secondary">
            PCA Components
            <input type="number" value={nComponents} onChange={(e) => setNComponents(+e.target.value)} className="ff-input w-24 block mt-1" min={1} />
          </label>
        </div>

        <LoadingButton onClick={handleTrain} loading={loading} loadingText="Training...">
          Train Model
        </LoadingButton>
      </div>

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
              <p className="text-xs text-fg-muted">Event ID</p>
              <p className="text-sm font-mono text-fg-secondary">{(result.event_id || "—").slice(0, 12)}</p>
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

/* ── Recognize Tab ── */
function RecognizeTab() {
  const [subjectId, setSubjectId] = useState("");
  const [modelId, setModelId] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleRecognize = () => {
    if (!subjectId.trim()) {
      setError("Subject ID is required");
      return;
    }
    setError("");
    setResult(null);
    setLoading(true);

    const params = new URLSearchParams();
    params.set("subject_id", subjectId.trim());
    if (modelId.trim()) params.set("model_id", modelId.trim());
    params.set("threshold", String(threshold));

    fetch(`/api/v1/recognition/predict?${params.toString()}`, { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => { setResult(data); setLoading(false); })
      .catch((e) => { setError(`Recognition error: ${e.message}`); setLoading(false); });
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">{error}</div>
      )}

      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Recognize Face</h3>
        <p className="text-xs text-fg-muted">Identify a subject using a trained model.</p>

        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">
            Subject ID
            <input
              type="text"
              value={subjectId}
              onChange={(e) => setSubjectId(e.target.value)}
              className="ff-input w-48 block mt-1"
              placeholder="e.g. subject_001"
            />
          </label>
          <label className="text-sm text-fg-secondary">
            Model ID
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="ff-input w-48 block mt-1"
              placeholder="Optional — uses latest"
            />
          </label>
        </div>

        <label className="text-sm text-fg-secondary block">
          Confidence Threshold: {threshold.toFixed(2)}
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(+e.target.value)}
            className="w-full mt-1"
          />
          <div className="flex justify-between text-xs text-fg-faint">
            <span>0</span>
            <span>1</span>
          </div>
        </label>

        <LoadingButton onClick={handleRecognize} loading={loading} loadingText="Recognizing...">
          Recognize
        </LoadingButton>
      </div>

      {result && (
        <div className="glass-card p-4 space-y-3 animate-fade-in">
          <h3 className="font-semibold text-fg-secondary">Predictions</h3>
          <div className="grid grid-cols-3 gap-4 mb-3">
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
              <p className="text-sm font-mono text-fg-secondary">{(result.model_id || "—").slice(0, 12)}</p>
            </div>
          </div>

          {/* Top-N predictions as ranked list with confidence bars */}
          {result.matches && result.matches.length > 0 && (
            <div className="space-y-1.5">
              {result.matches.map((m, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="text-xs text-fg-muted w-6 text-right">#{i + 1}</span>
                  <span className="text-sm text-fg-secondary font-mono w-32 truncate">{m.subject_id}</span>
                  <div className="flex-1 bg-surface rounded-full h-2.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${m.similarity >= 0.8 ? "bg-teal" : m.similarity >= 0.5 ? "bg-gold" : "bg-rose-ember"}`}
                      style={{ width: `${m.similarity * 100}%` }}
                    />
                  </div>
                  <span className={`text-sm font-semibold w-16 text-right ${m.similarity >= 0.8 ? "text-teal" : m.similarity >= 0.5 ? "text-gold" : "text-rose-ember"}`}>
                    {(m.similarity * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Models Tab ── */
function ModelsTab() {
  const [models, setModels] = useState<RecognitionModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const fetchModels = () => {
    setLoading(true);
    fetch("/api/v1/recognition/models")
      .then((r) => r.json())
      .then((data) => { setModels(Array.isArray(data) ? data : []); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  };

  useEffect(() => { fetchModels(); }, []);

  const deleteModel = (modelId: string) => {
    fetch(`/api/v1/recognition/models/${modelId}`, { method: "DELETE" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        setDeleteTarget(null);
        fetchModels();
      })
      .catch((e) => { setError(`Delete error: ${e.message}`); setDeleteTarget(null); });
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">{error}</div>
      )}

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Delete Model"
        message={`Are you sure you want to delete model "${deleteTarget?.slice(0, 16)}"?`}
        variant="danger"
        confirmLabel="Delete"
        onConfirm={() => deleteTarget && deleteModel(deleteTarget)}
        onCancel={() => setDeleteTarget(null)}
      />

      <div className="glass-card p-4">
        <h3 className="font-semibold text-fg-secondary mb-3">Model Registry</h3>
        {loading ? (
          <div className="space-y-1">{[...Array(3)].map((_, i) => <SkeletonRow key={i} />)}</div>
        ) : models.length === 0 ? (
          <EmptyState title="No models trained yet" subtitle="Train a PCA+SVM model from the Train tab" />
        ) : (
          <div className="space-y-2">
            {models.map((m) => (
              <div key={m.model_id} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <p className="text-sm text-fg-secondary font-mono">{m.model_id.slice(0, 16)}</p>
                    <span className="px-1.5 py-0.5 rounded text-xs bg-fg/5 text-fg-muted">{m.model_type}</span>
                  </div>
                  <p className="text-xs text-fg-faint">{m.num_classes} classes — {new Date(m.trained_at).toLocaleDateString()}</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 w-32">
                    <div className="flex-1 bg-surface-light rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${m.accuracy >= 0.9 ? "bg-teal" : m.accuracy >= 0.7 ? "bg-gold" : "bg-rose-ember"}`}
                        style={{ width: `${m.accuracy * 100}%` }}
                      />
                    </div>
                    <span className={`text-sm font-semibold tabular-nums ${m.accuracy >= 0.9 ? "text-teal" : m.accuracy >= 0.7 ? "text-gold" : "text-rose-ember"}`}>
                      {(m.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <button
                    onClick={() => setDeleteTarget(m.model_id)}
                    className="text-xs text-rose-ember hover:text-rose-ember/80 px-2 py-1 rounded hover:bg-rose-ember/10 transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Voice Tab ── */
function VoiceTab() {
  const [error, setError] = useState("");
  const [enrollFile, setEnrollFile] = useState<File | null>(null);
  const [enrollSubjectId, setEnrollSubjectId] = useState("");
  const [enrollResult, setEnrollResult] = useState<Record<string, unknown> | null>(null);
  const [enrolling, setEnrolling] = useState(false);
  const [verifyFile, setVerifyFile] = useState<File | null>(null);
  const [verifyTopK, setVerifyTopK] = useState(3);
  const [verifyResult, setVerifyResult] = useState<Record<string, unknown> | null>(null);
  const [verifying, setVerifying] = useState(false);

  const handleEnroll = () => {
    if (!enrollFile || !enrollSubjectId.trim()) { setError("Audio file and subject ID are required"); return; }
    setError(""); setEnrollResult(null); setEnrolling(true);
    const formData = new FormData();
    formData.append("audio", enrollFile);
    fetch(`/api/v1/recognition/voice/enroll?subject_id=${encodeURIComponent(enrollSubjectId.trim())}`, { method: "POST", body: formData })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => { setEnrollResult(data); setEnrolling(false); })
      .catch((e) => { setError(`Enroll error: ${e.message}`); setEnrolling(false); });
  };

  const handleVerify = () => {
    if (!verifyFile) { setError("Please select an audio file"); return; }
    setError(""); setVerifyResult(null); setVerifying(true);
    const formData = new FormData();
    formData.append("audio", verifyFile);
    fetch(`/api/v1/recognition/voice/verify?top_k=${verifyTopK}`, { method: "POST", body: formData })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => { setVerifyResult(data); setVerifying(false); })
      .catch((e) => { setError(`Verify error: ${e.message}`); setVerifying(false); });
  };

  return (
    <div className="space-y-4">
      {error && <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">{error}</div>}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Enroll Voice</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">Audio<input type="file" accept="audio/*,.wav,.pcm" onChange={(e) => setEnrollFile(e.target.files?.[0] || null)} className="ff-input block mt-1" /></label>
          <label className="text-sm text-fg-secondary">Subject ID<input type="text" value={enrollSubjectId} onChange={(e) => setEnrollSubjectId(e.target.value)} className="ff-input w-48 block mt-1" placeholder="e.g. speaker_001" /></label>
        </div>
        <LoadingButton onClick={handleEnroll} loading={enrolling} disabled={!enrollFile} className="btn-success" loadingText="Enrolling...">Enroll Voice</LoadingButton>
        {enrollResult && <div className="bg-teal/10 border border-teal/20 rounded-lg px-3 py-2 text-teal text-sm animate-fade-in">Voice enrolled — {String(enrollResult.subject_id || "success")}</div>}
      </div>
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Verify Voice</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">Audio<input type="file" accept="audio/*,.wav,.pcm" onChange={(e) => setVerifyFile(e.target.files?.[0] || null)} className="ff-input block mt-1" /></label>
          <label className="text-sm text-fg-secondary">Top K<input type="number" value={verifyTopK} onChange={(e) => setVerifyTopK(+e.target.value)} className="ff-input w-20 block mt-1" min={1} /></label>
        </div>
        <LoadingButton onClick={handleVerify} loading={verifying} disabled={!verifyFile} loadingText="Verifying...">Verify</LoadingButton>
        {verifyResult && Array.isArray(verifyResult.matches) && (
          <div className="space-y-1 mt-2">
            {(verifyResult.matches as Array<{ label: string; confidence: number }>).map((m, i) => (
              <div key={i} className="flex items-center justify-between bg-surface rounded-lg px-3 py-2">
                <span className="text-sm text-fg-secondary font-mono">{m.label}</span>
                <span className={`text-sm font-semibold ${m.confidence >= 0.8 ? "text-teal" : m.confidence >= 0.5 ? "text-gold" : "text-rose-ember"}`}>{(m.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Liveness Tab ── */
function LivenessTab() {
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState<{ is_live: boolean; score: number; checks: Record<string, number>; details: string; threshold: number } | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleCheck = () => {
    if (!file) { setError("Please select a face image"); return; }
    setError(""); setResult(null); setLoading(true);
    const formData = new FormData();
    formData.append("image", file);
    fetch(`/api/v1/recognition/liveness?threshold=${threshold}`, { method: "POST", body: formData })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => { setResult(data); setLoading(false); })
      .catch((e) => { setError(`Liveness error: ${e.message}`); setLoading(false); });
  };

  return (
    <div className="space-y-4">
      {error && <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">{error}</div>}
      <div className="glass-card p-4 space-y-3">
        <h3 className="font-semibold text-fg-secondary">Passive Liveness Detection</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="text-sm text-fg-secondary">Face Image<input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} className="ff-input block mt-1" /></label>
          <label className="text-sm text-fg-secondary">Threshold<input type="number" value={threshold} onChange={(e) => setThreshold(+e.target.value)} className="ff-input w-24 block mt-1" min={0} max={1} step={0.05} /></label>
        </div>
        <LoadingButton onClick={handleCheck} loading={loading} disabled={!file} loadingText="Checking...">Check Liveness</LoadingButton>
      </div>
      {result && (
        <div className="glass-card p-4 space-y-3 animate-fade-in">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold ${result.is_live ? "bg-teal/20 text-teal" : "bg-rose-ember/20 text-rose-ember"}`}>
              {result.is_live ? "\u2713" : "\u2717"}
            </div>
            <div>
              <p className={`text-lg font-bold ${result.is_live ? "text-teal" : "text-rose-ember"}`}>{result.is_live ? "Live" : "Spoof Detected"}</p>
              <p className="text-xs text-fg-muted">Score: {result.score.toFixed(3)} | Threshold: {result.threshold}</p>
            </div>
          </div>
          <div className="space-y-2">
            {Object.entries(result.checks).map(([name, score]) => (
              <div key={name} className="flex items-center gap-3">
                <span className="text-xs text-fg-secondary w-28 truncate">{name}</span>
                <div className="flex-1 bg-surface rounded-full h-2 overflow-hidden">
                  <div className={`h-full rounded-full ${score >= 0.5 ? "bg-teal" : "bg-rose-ember"}`} style={{ width: `${Math.min(score * 100, 100)}%` }} />
                </div>
                <span className="text-xs text-fg-muted w-12 text-right">{(score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
