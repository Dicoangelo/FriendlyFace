import { useState } from "react";
import { useNavigate } from "react-router-dom";

/* ---------- Live Verify Demo ---------- */

function VerifyDemo() {
  const [json, setJson] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");

  const handleVerify = () => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(json);
    } catch {
      setError("Invalid JSON — please paste a valid credential.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    fetch("/api/v1/seal/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential: parsed }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`Verification failed (${r.status})`);
        return r.json();
      })
      .then((data) => setResult(data as Record<string, unknown>))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="max-w-2xl mx-auto">
      <textarea
        value={json}
        onChange={(e) => setJson(e.target.value)}
        rows={5}
        className="w-full bg-surface border border-border-theme rounded-lg px-4 py-3 text-sm font-mono text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-2 focus:ring-cyan/40"
        placeholder='Paste a ForensicSeal credential JSON here...'
      />
      <div className="mt-3 flex justify-center">
        <button
          onClick={handleVerify}
          disabled={!json.trim() || loading}
          className="px-6 py-2.5 text-sm font-medium bg-cyan text-white rounded-lg hover:bg-cyan/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Verifying..." : "Verify Credential"}
        </button>
      </div>
      {error && (
        <div className="mt-4 bg-rose-ember/10 border border-rose-ember/30 text-rose-ember rounded-lg px-4 py-3 text-sm">
          {error}
        </div>
      )}
      {result && (
        <div className="mt-4 animate-fade-in">
          <pre className="bg-surface rounded-lg p-4 text-xs font-mono text-fg-secondary max-h-64 overflow-auto whitespace-pre-wrap border border-border-theme">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

/* ---------- Demo Request Form ---------- */

function DemoRequestForm() {
  const [form, setForm] = useState({ name: "", email: "", company: "", message: "" });
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
  };

  if (submitted) {
    return (
      <div className="text-center py-8">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-teal/10 text-teal mb-4">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h3 className="text-xl font-semibold text-fg mb-2">Request Received</h3>
        <p className="text-fg-muted">We will be in touch within 24 hours.</p>
      </div>
    );
  }

  const inputClass =
    "w-full bg-surface border border-border-theme rounded-lg px-4 py-2.5 text-sm text-fg-secondary placeholder:text-fg-faint/50 focus:outline-none focus:ring-2 focus:ring-cyan/40";

  return (
    <form onSubmit={handleSubmit} className="max-w-lg mx-auto space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-fg-muted mb-1.5">Name *</label>
          <input
            type="text"
            required
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className={inputClass}
            placeholder="Jane Smith"
          />
        </div>
        <div>
          <label className="block text-xs text-fg-muted mb-1.5">Email *</label>
          <input
            type="email"
            required
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
            className={inputClass}
            placeholder="jane@company.com"
          />
        </div>
      </div>
      <div>
        <label className="block text-xs text-fg-muted mb-1.5">Company</label>
        <input
          type="text"
          value={form.company}
          onChange={(e) => setForm({ ...form, company: e.target.value })}
          className={inputClass}
          placeholder="Acme Corp"
        />
      </div>
      <div>
        <label className="block text-xs text-fg-muted mb-1.5">Message</label>
        <textarea
          value={form.message}
          onChange={(e) => setForm({ ...form, message: e.target.value })}
          rows={3}
          className={inputClass}
          placeholder="Tell us about your AI system and compliance needs..."
        />
      </div>
      <div className="text-center pt-2">
        <button
          type="submit"
          className="px-8 py-3 text-sm font-medium bg-cyan text-white rounded-lg hover:bg-cyan/90 transition-colors"
        >
          Request a Demo
        </button>
      </div>
    </form>
  );
}

/* ---------- Section wrapper ---------- */

function Section({
  children,
  id,
  className = "",
}: {
  children: React.ReactNode;
  id?: string;
  className?: string;
}) {
  return (
    <section id={id} className={`max-w-6xl mx-auto px-6 py-20 ${className}`}>
      {children}
    </section>
  );
}

function SectionTitle({ children, sub }: { children: React.ReactNode; sub?: string }) {
  return (
    <div className="text-center mb-12">
      <h2 className="text-3xl font-bold text-fg mb-3">{children}</h2>
      {sub && <p className="text-fg-muted max-w-2xl mx-auto">{sub}</p>}
    </div>
  );
}

/* ---------- Main Page ---------- */

export default function SealLanding() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-page">
      {/* SEO meta tags are set via index.html or Helmet — hints for crawlers */}
      {/* Nav */}
      <nav className="border-b border-border-theme bg-sidebar/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <img src="/logo.png" alt="FriendlyFace" className="w-8 h-8" />
            <span className="text-xl font-bold gradient-text">FriendlyFace</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/landing")}
              className="text-sm text-fg-muted hover:text-fg transition-colors"
            >
              Platform
            </button>
            <button
              onClick={() => navigate("/pricing")}
              className="text-sm text-fg-muted hover:text-fg transition-colors"
            >
              Pricing
            </button>
            <button
              onClick={() => navigate("/login")}
              className="text-sm text-fg-muted hover:text-fg transition-colors"
            >
              Sign In
            </button>
            <button
              onClick={() => navigate("/signup")}
              className="px-4 py-2 text-sm font-medium bg-cyan text-white rounded-lg hover:bg-cyan/90 transition-colors"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* ===== 1. Hero ===== */}
      <section className="max-w-6xl mx-auto px-6 pt-24 pb-16 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-rose-ember/30 bg-rose-ember/10 text-rose-ember text-xs font-medium mb-8">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M12 3l9.66 16.59A1 1 0 0120.66 21H3.34a1 1 0 01-.86-1.41L12 3z" />
          </svg>
          EU AI Act enforcement begins August 2, 2026
        </div>
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-fg mb-6 leading-tight">
          The compliance certificate your
          <br />
          <span className="gradient-text">AI system needs before August 2, 2026</span>
        </h1>
        <p className="text-lg sm:text-xl text-fg-muted max-w-3xl mx-auto mb-10">
          ForensicSeal is a cryptographically signed, W3C Verifiable Credential that proves your AI system
          meets EU AI Act requirements. Automated. Tamper-proof. Court-admissible.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <a
            href="#demo-request"
            className="px-8 py-3 text-lg font-medium bg-cyan text-white rounded-lg hover:bg-cyan/90 transition-colors"
          >
            Get Your ForensicSeal
          </a>
          <a
            href="#how-it-works"
            className="px-8 py-3 text-lg font-medium border border-border-theme text-fg rounded-lg hover:bg-fg/5 transition-colors"
          >
            How It Works
          </a>
        </div>
      </section>

      {/* ===== 2. What is ForensicSeal ===== */}
      <Section id="what-is-forensicseal">
        <SectionTitle sub="A machine-verifiable proof that your AI system has been audited across six compliance layers.">
          What is ForensicSeal?
        </SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              title: "Cryptographically Signed",
              desc: "Ed25519 digital signatures ensure the certificate cannot be forged or tampered with after issuance.",
              icon: (
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              ),
            },
            {
              title: "W3C Verifiable Credential",
              desc: "Built on open standards. Any third party can independently verify your compliance status using the credential JSON.",
              icon: (
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              ),
            },
            {
              title: "6-Layer Assessment",
              desc: "Covers recognition accuracy, federated learning, forensic chain, fairness, explainability, and consent governance.",
              icon: (
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              ),
            },
          ].map((item) => (
            <div
              key={item.title}
              className="p-6 rounded-xl border border-border-theme bg-sidebar/50 hover:border-cyan/30 transition-colors"
            >
              <div className="text-cyan mb-4">{item.icon}</div>
              <h3 className="text-lg font-semibold text-fg mb-2">{item.title}</h3>
              <p className="text-sm text-fg-muted">{item.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* ===== 3. How it Works ===== */}
      <Section id="how-it-works" className="bg-sidebar/30">
        <SectionTitle sub="Three steps from connection to certification.">
          How It Works
        </SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            {
              step: "1",
              title: "Connect Your AI System",
              desc: "Integrate via API or upload your model artifacts. ForensicSeal connects to your inference pipeline, training data references, and governance docs.",
            },
            {
              step: "2",
              title: "Automated 6-Layer Compliance Check",
              desc: "Our engine runs bias audits, explainability analysis, consent verification, forensic chain validation, and federated learning checks automatically.",
            },
            {
              step: "3",
              title: "Receive Your Verifiable Certificate",
              desc: "Get a cryptographically signed W3C Verifiable Credential with per-layer scores, evidence links, and a public verification URL anyone can check.",
            },
          ].map((item) => (
            <div key={item.step} className="text-center">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-cyan/10 text-cyan text-2xl font-bold mb-4 border border-cyan/20">
                {item.step}
              </div>
              <h3 className="text-lg font-semibold text-fg mb-2">{item.title}</h3>
              <p className="text-sm text-fg-muted">{item.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* ===== 4. Why You Need It ===== */}
      <Section id="why-you-need-it">
        <SectionTitle sub="The EU AI Act is not optional. Non-compliance carries severe consequences.">
          Why You Need ForensicSeal
        </SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              value: "35M",
              unit: "EUR",
              label: "Maximum Fine",
              desc: "Or 7% of global annual turnover, whichever is higher. The EU AI Act imposes the largest AI-specific penalties in the world.",
              color: "text-rose-ember",
            },
            {
              value: "Aug 2",
              unit: "2026",
              label: "Enforcement Deadline",
              desc: "High-risk AI systems, including facial recognition, must demonstrate compliance by this date or face enforcement action.",
              color: "text-gold",
            },
            {
              value: "High",
              unit: "Risk",
              label: "Facial Recognition",
              desc: "Biometric identification systems are explicitly classified as high-risk under the EU AI Act, requiring conformity assessment.",
              color: "text-cyan",
            },
          ].map((item) => (
            <div
              key={item.label}
              className="p-6 rounded-xl border border-border-theme bg-sidebar/50 text-center"
            >
              <div className={`text-4xl font-bold ${item.color} mb-1`}>
                {item.unit === "EUR" ? "\u20AC" : ""}{item.value}
              </div>
              <div className="text-xs text-fg-faint uppercase tracking-wider mb-3">{item.unit === "EUR" ? "" : item.unit} {item.label}</div>
              <p className="text-sm text-fg-muted">{item.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* ===== 5. Live Verify Demo ===== */}
      <Section id="verify-demo" className="bg-sidebar/30">
        <SectionTitle sub="Paste any ForensicSeal credential JSON below to verify its authenticity in real-time.">
          Live Verification Demo
        </SectionTitle>
        <VerifyDemo />
      </Section>

      {/* ===== 6. Comparison Table ===== */}
      <Section id="comparison">
        <SectionTitle sub="See how ForensicSeal compares to traditional compliance approaches.">
          Competitive Comparison
        </SectionTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-border-theme">
                <th className="px-4 py-3 text-left text-xs font-semibold text-fg-muted uppercase tracking-wider">Criteria</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-cyan uppercase tracking-wider">ForensicSeal</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-fg-muted uppercase tracking-wider">Manual Audit</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-fg-faint uppercase tracking-wider">No Compliance</th>
              </tr>
            </thead>
            <tbody>
              {[
                { criteria: "Time to Certificate", seal: "Minutes", manual: "Weeks to months", none: "N/A" },
                { criteria: "Cost", seal: "From $99/mo", manual: "$10,000-50,000+", none: "Up to \u20AC35M fine" },
                { criteria: "Verifiability", seal: "Cryptographic (W3C VC)", manual: "PDF report", none: "None" },
                { criteria: "Automation", seal: "Fully automated", manual: "Manual review", none: "None" },
                { criteria: "Continuous Monitoring", seal: "Yes, with renewal", manual: "Point-in-time only", none: "None" },
                { criteria: "Court-Admissible", seal: "Yes (hash-chained)", manual: "Depends on auditor", none: "No" },
                { criteria: "Third-Party Verifiable", seal: "Yes (public URL)", manual: "On request", none: "No" },
              ].map((row) => (
                <tr key={row.criteria} className="border-b border-border-theme/50 hover:bg-fg/5 transition-colors">
                  <td className="px-4 py-3 text-fg-secondary font-medium">{row.criteria}</td>
                  <td className="px-4 py-3 text-center text-teal">{row.seal}</td>
                  <td className="px-4 py-3 text-center text-fg-muted">{row.manual}</td>
                  <td className="px-4 py-3 text-center text-fg-faint">{row.none}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* ===== 7. Pricing ===== */}
      <Section id="pricing" className="bg-sidebar/30">
        <SectionTitle sub="Choose the plan that fits your organization.">
          Pricing
        </SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          {[
            {
              name: "Starter",
              price: "$99",
              period: "/month",
              features: ["1 ForensicSeal certificate", "Quarterly compliance checks", "Basic compliance report", "Email support"],
              cta: "Get Started",
              href: "/signup",
              highlight: false,
            },
            {
              name: "Professional",
              price: "$299",
              period: "/month",
              features: ["10 ForensicSeal certificates", "Monthly compliance checks", "Full 6-layer assessment", "Compliance proxy", "Priority support"],
              cta: "Get Started",
              href: "/signup",
              highlight: true,
            },
            {
              name: "Enterprise",
              price: "$999",
              period: "/month",
              features: ["Unlimited ForensicSeal certificates", "Real-time compliance monitoring", "Conformity assessment (Annex IV)", "Dedicated support", "SLA guarantee", "On-premise option"],
              cta: "Contact Us",
              href: "#demo-request",
              highlight: false,
            },
          ].map((plan) => (
            <div
              key={plan.name}
              className={`p-6 rounded-xl border ${
                plan.highlight
                  ? "border-cyan/40 bg-cyan/5 ring-1 ring-cyan/20"
                  : "border-border-theme bg-sidebar/50"
              } flex flex-col`}
            >
              {plan.highlight && (
                <div className="text-xs text-cyan font-semibold uppercase tracking-wider mb-2">Most Popular</div>
              )}
              <h3 className="text-xl font-bold text-fg mb-1">{plan.name}</h3>
              <div className="mb-4">
                <span className="text-3xl font-bold text-fg">{plan.price}</span>
                <span className="text-fg-muted text-sm">{plan.period}</span>
              </div>
              <ul className="space-y-2 mb-6 flex-1">
                {plan.features.map((f) => (
                  <li key={f} className="flex items-start gap-2 text-sm text-fg-muted">
                    <svg className="w-4 h-4 text-teal flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    {f}
                  </li>
                ))}
              </ul>
              <button
                onClick={() => {
                  if (plan.href === "#demo-request") {
                    document.getElementById("demo-request")?.scrollIntoView({ behavior: "smooth" });
                  } else {
                    navigate(plan.href);
                  }
                }}
                className={`w-full py-2.5 text-sm font-medium rounded-lg transition-colors ${
                  plan.highlight
                    ? "bg-cyan text-white hover:bg-cyan/90"
                    : "border border-border-theme text-fg hover:bg-fg/5"
                }`}
              >
                {plan.cta}
              </button>
            </div>
          ))}
        </div>
      </Section>

      {/* ===== 8. Demo Request Form ===== */}
      <Section id="demo-request">
        <SectionTitle sub="Tell us about your AI system and we will schedule a personalized demo.">
          Request a Demo
        </SectionTitle>
        <DemoRequestForm />
      </Section>

      {/* Footer */}
      <footer className="border-t border-border-theme py-8">
        <div className="max-w-6xl mx-auto px-6 text-center text-sm text-fg-faint">
          FriendlyFace v0.1.0 -- ForensicSeal: AI Compliance Certificates
        </div>
      </footer>
    </div>
  );
}
