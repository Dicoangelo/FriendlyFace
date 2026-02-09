import { useEffect, useState } from "react";
import { SkeletonCard } from "../components/Skeleton";

interface ComplianceReport {
  compliant: boolean;
  overall_score: number;
  consent_coverage: number;
  bias_pass_rate: number;
  explanation_coverage: number;
  bundle_integrity: number;
  articles: ArticleAssessment[];
}

interface ArticleAssessment {
  article: string;
  title: string;
  status: string;
  details: string;
}

export default function Compliance() {
  const [report, setReport] = useState<ComplianceReport | null>(null);
  const [error, setError] = useState("");
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    fetch("/api/v1/governance/compliance")
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setReport)
      .catch((e) => setError(e.message));
  }, []);

  const generateReport = () => {
    setError("");
    setGenerating(true);
    fetch("/api/v1/governance/compliance/generate", { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        setReport(data);
        setGenerating(false);
      })
      .catch((e) => {
        setError(`Generate error: ${e.message}`);
        setGenerating(false);
      });
  };

  if (error && !report)
    return <div className="text-rose-ember">Error loading compliance: {error}</div>;

  if (!report)
    return (
      <div className="space-y-4">
        <SkeletonCard className="h-24" />
        <SkeletonCard className="h-24" />
        <SkeletonCard className="h-24" />
      </div>
    );

  const statusColor = report.compliant
    ? "bg-teal/10 text-teal border-teal/20"
    : "bg-rose-ember/10 text-rose-ember border-rose-ember/20";

  return (
    <div className="space-y-6">
      {/* Page title shown in header bar */}

      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm">
          {error}
        </div>
      )}

      {/* Status banner */}
      <div className={`rounded-lg border-2 p-4 ${statusColor}`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-lg font-bold uppercase">
              {report.compliant ? "Compliant" : "Non-Compliant"}
            </p>
            <p className="text-sm">
              Overall score: {report.overall_score != null && isFinite(report.overall_score) ? `${(report.overall_score * 100).toFixed(1)}%` : "N/A"}
            </p>
          </div>
          <button
            onClick={generateReport}
            disabled={generating}
            className="btn-primary"
          >
            {generating ? "Generating..." : "Regenerate"}
          </button>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Consent Coverage" value={report.consent_coverage} />
        <MetricCard label="Bias Pass Rate" value={report.bias_pass_rate} />
        <MetricCard label="Explanation Coverage" value={report.explanation_coverage} />
        <MetricCard label="Bundle Integrity" value={report.bundle_integrity} />
      </div>

      {/* Article assessments */}
      {report.articles && report.articles.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-fg-secondary">Article Assessments</h3>
          {report.articles.map((a) => {
            const artColor =
              a.status === "pass"
                ? "border-teal/20"
                : a.status === "warning"
                  ? "border-gold/20"
                  : "border-rose-ember/20";
            const badgeColor =
              a.status === "pass"
                ? "bg-teal/10 text-teal"
                : a.status === "warning"
                  ? "bg-gold/10 text-gold"
                  : "bg-rose-ember/10 text-rose-ember";
            return (
              <div key={a.article} className={`glass-card p-4 border-l-2 ${artColor}`}>
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <p className="font-semibold text-fg-secondary">{a.article}</p>
                    <p className="text-sm text-fg-muted">{a.title}</p>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${badgeColor}`}>
                    {a.status.toUpperCase()}
                  </span>
                </div>
                <p className="text-xs text-fg-muted">{a.details}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: number | null | undefined }) {
  const hasValue = value != null && isFinite(value);
  const pct = hasValue ? (value * 100).toFixed(1) : "—";
  const color = !hasValue ? "fg-faint" : value >= 0.8 ? "teal" : value >= 0.5 ? "gold" : "rose-ember";
  return (
    <div className={`glass-card p-4 border-l-2 border-${color}/20`}>
      <p className="text-sm text-fg-muted">{label}</p>
      <p className={`text-2xl font-bold mt-1 text-${color}`}>{pct}{hasValue ? "%" : ""}</p>
    </div>
  );
}
