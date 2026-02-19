import { useNavigate } from "react-router-dom";

const FEATURES = [
  {
    title: "Forensic-Grade Recognition",
    description: "PCA + SVM facial recognition with hash-chained audit trails and Merkle verification.",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
  },
  {
    title: "Federated Learning",
    description: "Privacy-preserving distributed training with differential privacy and poisoning detection.",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m-2 6h2m14-6h2m-2 6h2M7 7h10v10H7V7z" />
      </svg>
    ),
  },
  {
    title: "Zero-Knowledge Proofs",
    description: "Schnorr ZK proofs for privacy-preserving verification of forensic evidence.",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
  },
  {
    title: "EU AI Act Compliance",
    description: "Built-in OSCAL export, bias auditing, and explainability for regulatory compliance.",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
      </svg>
    ),
  },
  {
    title: "Decentralized Identity",
    description: "Ed25519 DID:key identifiers and W3C Verifiable Credentials for forensic actors.",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
      </svg>
    ),
  },
  {
    title: "Real-Time Explainability",
    description: "LIME, SHAP, and SDD saliency maps for every inference decision.",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
  },
];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-page">
      {/* Nav */}
      <nav className="border-b border-border-theme bg-sidebar/50 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <img src="/logo.png" alt="FriendlyFace" className="w-8 h-8" />
            <span className="text-xl font-bold gradient-text">FriendlyFace</span>
          </div>
          <div className="flex items-center gap-4">
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

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 py-24 text-center">
        <h1 className="text-5xl font-bold text-fg mb-6 leading-tight">
          Forensic AI Recognition
          <br />
          <span className="gradient-text">Built for Compliance</span>
        </h1>
        <p className="text-xl text-fg-muted max-w-2xl mx-auto mb-10">
          The only facial recognition platform with hash-chained audit trails, zero-knowledge proofs,
          and EU AI Act compliance built in. ICDF2C 2024 schema.
        </p>
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={() => navigate("/signup")}
            className="px-8 py-3 text-lg font-medium bg-cyan text-white rounded-lg hover:bg-cyan/90 transition-colors"
          >
            Start Free Trial
          </button>
          <button
            onClick={() => navigate("/pricing")}
            className="px-8 py-3 text-lg font-medium border border-border-theme text-fg rounded-lg hover:bg-fg/5 transition-colors"
          >
            View Pricing
          </button>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="text-3xl font-bold text-fg text-center mb-12">
          Six-Layer Forensic Architecture
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {FEATURES.map((feature) => (
            <div
              key={feature.title}
              className="p-6 rounded-xl border border-border-theme bg-sidebar/50 hover:border-cyan/30 transition-colors"
            >
              <div className="text-cyan mb-4">{feature.icon}</div>
              <h3 className="text-lg font-semibold text-fg mb-2">{feature.title}</h3>
              <p className="text-sm text-fg-muted">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-6xl mx-auto px-6 py-20 text-center">
        <div className="p-12 rounded-2xl border border-border-theme bg-sidebar/50">
          <h2 className="text-3xl font-bold text-fg mb-4">Ready to get started?</h2>
          <p className="text-fg-muted mb-8 max-w-xl mx-auto">
            Deploy forensic-grade AI recognition with full compliance in minutes.
          </p>
          <button
            onClick={() => navigate("/signup")}
            className="px-8 py-3 text-lg font-medium bg-cyan text-white rounded-lg hover:bg-cyan/90 transition-colors"
          >
            Create Your Account
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border-theme py-8">
        <div className="max-w-6xl mx-auto px-6 text-center text-sm text-fg-faint">
          FriendlyFace v0.1.0 -- Forensic-Friendly AI Recognition
        </div>
      </footer>
    </div>
  );
}
