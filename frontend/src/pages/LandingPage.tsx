import { useNavigate } from "react-router-dom";
import ParticleBackground from "../components/ParticleBackground";
import AnimatedCounter from "../components/AnimatedCounter";
import TerminalDemo from "../components/TerminalDemo";

const FEATURES = [
  {
    title: "Forensic-Grade Recognition",
    description:
      "PCA + SVM facial recognition with hash-chained audit trails, voice biometrics, and multi-modal fusion.",
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
    gradient: "from-cyan/20 to-teal/10",
    border: "group-hover:border-cyan/40",
  },
  {
    title: "Federated Learning",
    description:
      "Privacy-preserving distributed training with DP-FedAvg, epsilon/delta tracking, and poisoning detection.",
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m-2 6h2m14-6h2m-2 6h2M7 7h10v10H7V7z" />
      </svg>
    ),
    gradient: "from-amethyst/20 to-pink-500/10",
    border: "group-hover:border-amethyst/40",
  },
  {
    title: "Zero-Knowledge Proofs",
    description:
      "Schnorr ZK proofs with Fiat-Shamir heuristic for privacy-preserving evidence verification.",
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
    gradient: "from-teal/20 to-cyan/10",
    border: "group-hover:border-teal/40",
  },
  {
    title: "EU AI Act Compliance",
    description:
      "Built-in OSCAL export, demographic parity auditing, equalized odds checks, and automated compliance reports.",
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
      </svg>
    ),
    gradient: "from-gold/20 to-amber-500/10",
    border: "group-hover:border-gold/40",
  },
  {
    title: "Decentralized Identity",
    description:
      "Ed25519 DID:key identifiers and W3C Verifiable Credentials with PyNaCl/libsodium cryptography.",
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
      </svg>
    ),
    gradient: "from-pink-500/20 to-amethyst/10",
    border: "group-hover:border-pink-500/40",
  },
  {
    title: "Real-Time Explainability",
    description:
      "LIME saliency, KernelSHAP values, and SDD 7-region decomposition for every inference decision.",
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
    gradient: "from-cyan/20 to-amethyst/10",
    border: "group-hover:border-cyan/40",
  },
];

const LAYERS = [
  { num: 6, name: "Consent & Governance", desc: "Consent Engine • Compliance Reporter • EU AI Act Checks", color: "bg-gold/80", glow: "shadow-gold/20" },
  { num: 5, name: "Explainability (XAI)", desc: "LIME Saliency • KernelSHAP • SDD Saliency Maps", color: "bg-amethyst/80", glow: "shadow-amethyst/20" },
  { num: 4, name: "Fairness & Bias Auditor", desc: "Demographic Parity • Equalized Odds • Auto-Audit", color: "bg-pink-500/80", glow: "shadow-pink-500/20" },
  { num: 3, name: "Blockchain Forensic Layer", desc: "Hash-Chained Events • Merkle Tree • Provenance DAG • ZK/DID", color: "bg-cyan/80", glow: "shadow-cyan/20" },
  { num: 2, name: "Federated Learning Engine", desc: "FedAvg + DP-FedAvg • Poisoning Detection • Privacy Budgets", color: "bg-teal/80", glow: "shadow-teal/20" },
  { num: 1, name: "Recognition Engine", desc: "PCA + SVM • Voice Biometrics (MFCC) • Multi-Modal Fusion", color: "bg-cyan/60", glow: "shadow-cyan/10" },
];

const STATS = [
  { value: 84, suffix: "+", label: "API Endpoints" },
  { value: 6, suffix: "", label: "Forensic Layers" },
  { value: 1345, suffix: "+", label: "Passing Tests" },
  { value: 93, suffix: "%", label: "Test Coverage" },
];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-page">
      {/* ── Nav ─────────────────────────────────────────── */}
      <nav className="sticky top-0 z-50 border-b border-border-theme bg-sidebar/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <img src="/logo.png" alt="FriendlyFace" className="w-8 h-8" />
            <span className="text-xl font-bold gradient-text tracking-tight">FriendlyFace</span>
          </div>
          <div className="flex items-center gap-2 sm:gap-4">
            <a
              href="https://github.com/Dicoangelo/FriendlyFace"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-1.5 text-sm text-fg-muted hover:text-fg transition-colors"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" /></svg>
              GitHub
            </a>
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
              className="btn-primary text-sm !py-2"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* ── Hero ────────────────────────────────────────── */}
      <section className="relative overflow-hidden">
        <ParticleBackground />
        <div className="relative max-w-7xl mx-auto px-6 pt-20 pb-24 md:pt-28 md:pb-32 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 mb-8 rounded-full border border-cyan/20 bg-cyan/5 text-sm text-cyan animate-fade-in">
            <span className="w-2 h-2 rounded-full bg-teal animate-pulse" />
            Built on ICDF2C 2024 Research — University of Windsor
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-fg mb-6 leading-[1.1] tracking-tight animate-fade-in">
            Forensic AI Recognition
            <br />
            <span className="gradient-text">Built for Compliance</span>
          </h1>

          <p className="text-lg sm:text-xl text-fg-muted max-w-2xl mx-auto mb-10 leading-relaxed animate-fade-in">
            The only facial recognition platform with hash-chained audit trails,
            zero-knowledge proofs, and EU AI Act compliance built in.
            Every inference is Merkle-verified and courtroom-ready.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in">
            <button
              onClick={() => navigate("/signup")}
              className="btn-primary text-lg !px-8 !py-3 w-full sm:w-auto"
            >
              Start Free Trial
            </button>
            <button
              onClick={() => navigate("/")}
              className="btn-ghost text-lg !px-8 !py-3 w-full sm:w-auto"
            >
              Explore Dashboard →
            </button>
          </div>
        </div>
      </section>

      {/* ── Stats ───────────────────────────────────────── */}
      <section className="border-y border-border-theme bg-sidebar/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
            {STATS.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-3xl sm:text-4xl font-bold gradient-text mb-1">
                  <AnimatedCounter end={stat.value} suffix={stat.suffix} />
                </div>
                <p className="text-sm text-fg-muted">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Architecture ────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-20 md:py-28">
        <div className="text-center mb-14">
          <h2 className="text-3xl sm:text-4xl font-bold text-fg mb-4">
            Six-Layer Forensic Architecture
          </h2>
          <p className="text-fg-muted max-w-xl mx-auto">
            Every operation across all 6 layers produces an immutable ForensicEvent
            linked into a hash chain, Merkle tree, and provenance DAG.
          </p>
        </div>

        <div className="max-w-3xl mx-auto space-y-2">
          {LAYERS.map((layer, i) => (
            <div
              key={layer.num}
              className="group glass-card p-4 flex items-center gap-4 transition-all duration-300 hover:scale-[1.02] cursor-default"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className={`flex-shrink-0 w-10 h-10 rounded-lg ${layer.color} flex items-center justify-center text-white font-bold text-sm shadow-lg ${layer.glow}`}>
                L{layer.num}
              </div>
              <div className="min-w-0">
                <h3 className="font-semibold text-fg text-sm sm:text-base">{layer.name}</h3>
                <p className="text-xs sm:text-sm text-fg-muted truncate">{layer.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ────────────────────────────────────── */}
      <section className="bg-sidebar/30 border-y border-border-theme">
        <div className="max-w-7xl mx-auto px-6 py-20 md:py-28">
          <div className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-fg mb-4">
              Everything You Need
            </h2>
            <p className="text-fg-muted max-w-xl mx-auto">
              From recognition to compliance — a complete forensic AI platform.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((feature) => (
              <div
                key={feature.title}
                className={`group glass-card p-6 transition-all duration-300 hover:scale-[1.02] border ${feature.border} border-transparent`}
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center text-fg mb-4 transition-transform duration-300 group-hover:scale-110`}>
                  {feature.icon}
                </div>
                <h3 className="text-base font-semibold text-fg mb-2">{feature.title}</h3>
                <p className="text-sm text-fg-muted leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Terminal Demo ───────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-20 md:py-28">
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold text-fg mb-4">
            See It in Action
          </h2>
          <p className="text-fg-muted max-w-xl mx-auto">
            84+ API endpoints ready to go. Hash-chained, Merkle-verified, bias-audited.
          </p>
        </div>
        <TerminalDemo />
      </section>

      {/* ── Research Lineage ────────────────────────────── */}
      <section className="bg-sidebar/30 border-y border-border-theme">
        <div className="max-w-7xl mx-auto px-6 py-16 md:py-20">
          <div className="glass-gold p-8 md:p-12 text-center">
            <p className="text-xs uppercase tracking-widest text-gold mb-3 font-semibold">Research Lineage</p>
            <h3 className="text-2xl sm:text-3xl font-bold text-fg mb-4">
              Built on Peer-Reviewed Science
            </h3>
            <p className="text-fg-muted max-w-2xl mx-auto mb-6 leading-relaxed">
              Implements Safiia Mohammed's forensic-friendly AI framework from the
              <strong className="text-fg"> University of Windsor</strong> (ICDF2C 2024) with state-of-the-art
              2025–2026 components including Ed25519 DIDs, Schnorr ZK proofs, DP-FedAvg,
              and SDD saliency maps.
            </p>
            <div className="flex flex-wrap justify-center gap-3">
              {["ICDF2C 2024", "EU AI Act", "W3C DID", "W3C VC", "OSCAL", "GDPR Art. 17"].map((tag) => (
                <span key={tag} className="px-3 py-1 rounded-full text-xs font-medium border border-gold/30 text-gold bg-gold/5">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-20 md:py-28 text-center">
        <div className="border-sovereign p-10 md:p-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-fg mb-4">
            Ready to deploy forensic-grade AI?
          </h2>
          <p className="text-fg-muted mb-8 max-w-xl mx-auto">
            Full compliance in minutes. Open source, self-hostable,
            Docker-ready.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigate("/signup")}
              className="btn-primary text-lg !px-8 !py-3 w-full sm:w-auto"
            >
              Create Your Account
            </button>
            <a
              href="https://github.com/Dicoangelo/FriendlyFace"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-ghost text-lg !px-8 !py-3 w-full sm:w-auto inline-flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" /></svg>
              View on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────── */}
      <footer className="border-t border-border-theme bg-sidebar/50">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <img src="/logo.png" alt="FriendlyFace" className="w-6 h-6 opacity-60" />
              <span className="text-sm text-fg-faint">FriendlyFace v0.1.0 — Forensic-Friendly AI Recognition</span>
            </div>
            <div className="flex items-center gap-6 text-sm text-fg-faint">
              <a href="https://github.com/Dicoangelo/FriendlyFace" target="_blank" rel="noopener noreferrer" className="hover:text-fg transition-colors">GitHub</a>
              <button onClick={() => navigate("/pricing")} className="hover:text-fg transition-colors">Pricing</button>
              <button onClick={() => navigate("/")} className="hover:text-fg transition-colors">Dashboard</button>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
