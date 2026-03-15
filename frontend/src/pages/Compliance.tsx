import { useState } from "react";
import { useFetch } from "../hooks/useFetch";
import { SkeletonCard } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";
import LoadingButton from "../components/LoadingButton";
import ProgressRing from "../components/ProgressRing";

/* ---------- types matching API response from ComplianceReporter ---------- */

interface ArticleSection {
  title: string;
  description: string;
  status: string;
  consent_coverage?: MetricStats;
  bias_audit?: MetricStats;
  explanation_coverage?: MetricStats;
  bundle_integrity?: MetricStats;
}

interface MetricStats {
  coverage_pct?: number;
  pass_rate_pct?: number;
  integrity_pct?: number;
  total?: number;
  covered?: number;
  passed?: number;
  failed?: number;
  verified?: number;
}

interface ComplianceReport {
  report_id: string;
  generated_at: string;
  compliant: boolean;
  overall_compliance_score: number;
  article_5: ArticleSection;
  article_14: ArticleSection;
  metrics: {
    consent_coverage_pct: number;
    bias_audit_pass_rate_pct: number;
    explanation_coverage_pct: number;
    bundle_integrity_pct: number;
  };
  event_id?: string;
}

/* ---------- constants ---------- */

const SECTIONS: {
  key: "data_governance" | "transparency" | "human_oversight" | "robustness";
  label: string;
  requirements: {
    name: string;
    metricKey: keyof ComplianceReport["metrics"];
    articleKey: "article_5" | "article_14";
    details: string;
  }[];
}[] = [
  {
    key: "data_governance",
    label: "Data Governance",
    requirements: [
      {
        name: "Consent Coverage",
        metricKey: "consent_coverage_pct",
        articleKey: "article_5",
        details:
          "Percentage of subjects with valid, active consent records. Requires >= 70% for compliance.",
      },
    ],
  },
  {
    key: "transparency",
    label: "Transparency",
    requirements: [
      {
        name: "Explanation Coverage",
        metricKey: "explanation_coverage_pct",
        articleKey: "article_14",
        details:
          "Percentage of inference events with associated LIME/SHAP/SDD explanations. Requires >= 70%.",
      },
    ],
  },
  {
    key: "human_oversight",
    label: "Human Oversight",
    requirements: [
      {
        name: "Bias Audit Pass Rate",
        metricKey: "bias_audit_pass_rate_pct",
        articleKey: "article_5",
        details:
          "Percentage of bias audits passing fairness thresholds (demographic parity + equalized odds). Requires >= 70%.",
      },
    ],
  },
  {
    key: "robustness",
    label: "Robustness",
    requirements: [
      {
        name: "Bundle Integrity",
        metricKey: "bundle_integrity_pct",
        articleKey: "article_14",
        details:
          "Percentage of forensic bundles with verified hash chains, Merkle proofs, and ZK proofs. Requires >= 70%.",
      },
    ],
  },
];

const EU_AI_ACT_ARTICLES = [
  { id: "Article 5", title: "Prohibited Practices", key: "article_5" as const },
  { id: "Article 14", title: "Human Oversight", key: "article_14" as const },
];

/* ---------- helpers ---------- */

function statusBadge(status: string) {
  if (status === "pass")
    return (
      <span className="px-2 py-0.5 rounded text-xs font-medium bg-teal/10 text-teal">
        PASS
      </span>
    );
  if (status === "partial" || status === "warning")
    return (
      <span className="px-2 py-0.5 rounded text-xs font-medium bg-gold/10 text-gold">
        PARTIAL
      </span>
    );
  return (
    <span className="px-2 py-0.5 rounded text-xs font-medium bg-rose-ember/10 text-rose-ember">
      FAIL
    </span>
  );
}

function scoreBar(pct: number) {
  const color = pct >= 70 ? "bg-teal" : pct >= 50 ? "bg-gold" : "bg-rose-ember";
  return (
    <div className="flex items-center gap-2 flex-1 min-w-0">
      <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <span className="text-xs text-fg-muted w-12 text-right">{pct.toFixed(1)}%</span>
    </div>
  );
}

function requirementStatus(pct: number): string {
  if (pct >= 70) return "pass";
  if (pct >= 50) return "partial";
  return "fail";
}

/* ---------- component ---------- */

export default function Compliance() {
  const {
    data: report,
    loading,
    error: fetchError,
    retry,
  } = useFetch<ComplianceReport>("/api/v1/governance/compliance");

  const [error, setError] = useState("");
  const [rechecking, setRechecking] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const toggleSection = (key: string) =>
    setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }));

  /* Re-check compliance */
  const recheck = () => {
    setError("");
    setRechecking(true);
    fetch("/api/v1/governance/compliance/generate", { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(() => {
        retry();
        setRechecking(false);
      })
      .catch((e) => {
        setError(`Re-check failed: ${e.message}`);
        setRechecking(false);
      });
  };

  /* OSCAL export */
  const exportOscal = () => {
    setExporting(true);
    setError("");
    fetch("/api/v1/governance/compliance/export")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "friendlyface-oscal-export.json";
        a.click();
        URL.revokeObjectURL(url);
        setExporting(false);
      })
      .catch((e) => {
        setError(`OSCAL export failed: ${e.message}`);
        setExporting(false);
      });
  };

  /* ---------- loading skeleton ---------- */
  if (loading && !report) {
    return (
      <div className="space-y-6">
        <SkeletonCard className="h-32" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SkeletonCard className="h-24" />
          <SkeletonCard className="h-24" />
          <SkeletonCard className="h-24" />
          <SkeletonCard className="h-24" />
        </div>
      </div>
    );
  }

  /* ---------- empty state ---------- */
  if (!report && !fetchError && !error) {
    return (
      <EmptyState
        title="No compliance data yet"
        subtitle="Run a compliance check to generate your first report."
        action={
          <LoadingButton onClick={recheck} loading={rechecking} loadingText="Checking...">
            Run Compliance Check
          </LoadingButton>
        }
      />
    );
  }

  /* ---------- error-only state ---------- */
  if (!report) {
    return (
      <div className="space-y-4">
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-3 text-rose-ember text-sm flex items-center justify-between">
          <span>Failed to load compliance data: {fetchError || error}</span>
          <button onClick={retry} className="btn-ghost text-xs ml-4">
            Retry
          </button>
        </div>
      </div>
    );
  }

  /* ---------- main render ---------- */
  const score = report.overall_compliance_score / 100; // 0-1 for ProgressRing
  const scoreColor = report.compliant ? "text-teal" : "text-rose-ember";
  const riskLevel = score >= 0.9 ? "Minimal" : score >= 0.7 ? "Limited" : score >= 0.5 ? "High" : "Unacceptable";
  const riskColor =
    riskLevel === "Minimal"
      ? "bg-teal/10 text-teal"
      : riskLevel === "Limited"
        ? "bg-cyan/10 text-cyan"
        : riskLevel === "High"
          ? "bg-gold/10 text-gold"
          : "bg-rose-ember/10 text-rose-ember";

  return (
    <div className="space-y-6">
      {/* Error banner */}
      {error && (
        <div className="bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError("")} className="ml-2 text-rose-ember hover:text-rose-ember/70">
            &times;
          </button>
        </div>
      )}

      {/* Overall compliance ring + badge + actions */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-6 flex-wrap">
          <ProgressRing
            value={score}
            size={96}
            strokeWidth={8}
            color={scoreColor}
            label={`${report.overall_compliance_score.toFixed(0)}%`}
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-xl font-bold text-fg-secondary">
                Overall Compliance
              </h2>
              <span
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  report.compliant
                    ? "bg-teal/10 text-teal"
                    : "bg-rose-ember/10 text-rose-ember"
                }`}
              >
                {report.compliant ? "COMPLIANT" : "NON-COMPLIANT"}
              </span>
            </div>
            <p className="text-sm text-fg-muted">
              Score: {report.overall_compliance_score.toFixed(1)}% &middot; Generated{" "}
              {new Date(report.generated_at).toLocaleString()}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <LoadingButton
              onClick={recheck}
              loading={rechecking}
              loadingText="Checking..."
            >
              Re-check Compliance
            </LoadingButton>
            <LoadingButton
              onClick={exportOscal}
              loading={exporting}
              loadingText="Exporting..."
              className="btn-ghost"
            >
              Export OSCAL
            </LoadingButton>
          </div>
        </div>
      </div>

      {/* Requirement sections — collapsible cards */}
      {SECTIONS.map((section) => {
        const isCollapsed = collapsed[section.key] ?? false;
        return (
          <div key={section.key} className="glass-card overflow-hidden">
            <button
              onClick={() => toggleSection(section.key)}
              className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-surface/50 transition-colors"
            >
              <h3 className="font-semibold text-fg-secondary">{section.label}</h3>
              <svg
                className={`w-5 h-5 text-fg-muted transition-transform ${isCollapsed ? "" : "rotate-180"}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {!isCollapsed && (
              <div className="px-5 pb-4 space-y-3">
                {section.requirements.map((req) => {
                  const pct = report.metrics[req.metricKey];
                  const status = requirementStatus(pct);
                  return (
                    <div
                      key={req.name}
                      className="flex items-center gap-4 py-2 border-t border-fg-faint/10"
                    >
                      <span className="text-sm font-medium text-fg-secondary w-48 shrink-0">
                        {req.name}
                      </span>
                      {statusBadge(status)}
                      {scoreBar(pct)}
                      <span className="text-xs text-fg-muted max-w-xs hidden lg:block">
                        {req.details}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}

      {/* EU AI Act Summary */}
      <div className="glass-card p-5">
        <h3 className="font-semibold text-fg-secondary mb-3">EU AI Act Summary</h3>
        <div className="flex items-center gap-3 mb-4">
          <span className="text-sm text-fg-muted">Risk Level:</span>
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${riskColor}`}>
            {riskLevel.toUpperCase()}
          </span>
        </div>
        <div className="space-y-2">
          <p className="text-xs text-fg-muted font-medium uppercase tracking-wide mb-2">
            Applicable Articles
          </p>
          {EU_AI_ACT_ARTICLES.map((art) => {
            const section = report[art.key];
            return (
              <div
                key={art.id}
                className="flex items-center justify-between py-2 border-t border-fg-faint/10"
              >
                <div>
                  <span className="text-sm font-medium text-fg-secondary">
                    {art.id}
                  </span>
                  <span className="text-sm text-fg-muted ml-2">
                    &mdash; {art.title}
                  </span>
                </div>
                {statusBadge(section.status)}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
