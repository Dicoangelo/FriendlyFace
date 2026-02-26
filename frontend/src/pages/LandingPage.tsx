import { useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";

const FEATURES = [
  {
    title: "Forensic-Grade Recognition",
    description: "PCA + SVM facial recognition with hash-chained audit trails and Merkle verification.",
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
    gradient: "from-cyan/20 to-cyan/5",
    iconColor: "text-cyan",
  },
  {
    title: "Federated Learning",
    description: "Privacy-preserving distributed training with differential privacy and poisoning detection.",
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m-2 6h2m14-6h2m-2 6h2M7 7h10v10H7V7z" />
      </svg>
    ),
    gradient: "from-amethyst/20 to-amethyst/5",
    iconColor: "text-amethyst",
  },
  {
    title: "Zero-Knowledge Proofs",
    description: "Schnorr ZK proofs for privacy-preserving verification of forensic evidence.",
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
    gradient: "from-teal/20 to-teal/5",
    iconColor: "text-teal",
  },
  {
    title: "EU AI Act Compliance",
    description: "Built-in OSCAL export, bias auditing, and explainability for regulatory compliance.",
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
      </svg>
    ),
    gradient: "from-gold/20 to-gold/5",
    iconColor: "text-gold",
  },
  {
    title: "Decentralized Identity",
    description: "Ed25519 DID:key identifiers and W3C Verifiable Credentials for forensic actors.",
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
      </svg>
    ),
    gradient: "from-magenta/20 to-magenta/5",
    iconColor: "text-magenta",
  },
  {
    title: "Real-Time Explainability",
    description: "LIME, SHAP, and SDD saliency maps for every inference decision.",
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
    gradient: "from-holo-blue/20 to-holo-blue/5",
    iconColor: "text-holo-blue",
  },
];

const STATS = [
  { label: "Forensic Events Tracked", value: 50000, suffix: "+" },
  { label: "Audit Compliance Rate", value: 99.9, suffix: "%" },
  { label: "ZK Proofs Generated", value: 12000, suffix: "+" },
  { label: "Uptime SLA", value: 99.99, suffix: "%" },
];

const TRUST_LOGOS = [
  "ICDF2C 2024",
  "EU AI Act",
  "GDPR",
  "W3C DID",
  "OSCAL",
];

function AnimatedCounter({ target, suffix = "" }: { target: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true;
          const duration = 2000;
          const steps = 60;
          const increment = target / steps;
          let current = 0;
          const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
              setCount(target);
              clearInterval(timer);
            } else {
              setCount(Math.floor(current * 10) / 10);
            }
          }, duration / steps);
        }
      },
      { threshold: 0.3 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [target]);

  const decimalPlaces = target.toString().split(".")[1]?.length || 0;
  const display = decimalPlaces > 0
    ? count.toFixed(decimalPlaces)
    : Math.floor(count).toLocaleString();

  return (
    <div ref={ref} className="text-3xl md:text-4xl font-bold text-fg tabular-nums">
      {display}
      <span className="text-cyan">{suffix}</span>
    </div>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <div className="min-h-screen bg-page overflow-hidden">
      {/* Nav — fixed with scroll blur */}
      <nav className={`fixed top-0 inset-x-0 z-50 transition-all duration-300 ${scrolled ? "bg-sidebar/80 backdrop-blur-md border-b border-border-theme shadow-sm" : "bg-transparent"}`}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <img src="/logo.png" alt="FriendlyFace" className="w-8 h-8" />
            <span className="text-xl font-bold gradient-text">FriendlyFace</span>
          </div>
          <div className="flex items-center gap-2 sm:gap-4">
            <button
              onClick={() => navigate("/pricing")}
              className="text-sm text-fg-muted hover:text-fg transition-colors hidden sm:inline-block"
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
              className="btn-primary"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative max-w-6xl mx-auto px-6 pt-32 pb-20 text-center">
        {/* Animated gradient orbs */}
        <div className="absolute top-10 left-1/4 w-72 h-72 bg-amethyst/20 rounded-full blur-[120px] animate-pulse pointer-events-none" />
        <div className="absolute top-20 right-1/4 w-80 h-80 bg-cyan/15 rounded-full blur-[120px] animate-pulse pointer-events-none" style={{ animationDelay: "1s" }} />
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-96 h-48 bg-magenta/10 rounded-full blur-[100px] pointer-events-none" />

        <div className="relative">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-border-theme bg-surface/50 text-xs font-medium text-fg-secondary mb-8">
            <span className="w-2 h-2 rounded-full bg-teal animate-pulse" />
            ICDF2C 2024 Reference Implementation
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-fg mb-6 leading-tight tracking-tight">
            Forensic AI Recognition
            <br />
            <span className="gradient-text">Built for Compliance</span>
          </h1>
          <p className="text-lg sm:text-xl text-fg-muted max-w-2xl mx-auto mb-10 leading-relaxed">
            The only facial recognition platform with hash-chained audit trails, zero-knowledge proofs,
            and EU AI Act compliance built in.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigate("/signup")}
              className="btn-primary px-8 py-3 text-base"
            >
              Start Free Trial
              <svg className="w-4 h-4 ml-2 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
              </svg>
            </button>
            <button
              onClick={() => navigate("/pricing")}
              className="btn-ghost px-8 py-3 text-base"
            >
              View Pricing
            </button>
          </div>
        </div>
      </section>

      {/* Trust bar */}
      <section className="border-y border-border-theme bg-surface/30">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-10">
            {TRUST_LOGOS.map((name) => (
              <span
                key={name}
                className="text-xs sm:text-sm font-semibold text-fg-faint uppercase tracking-wider"
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="max-w-6xl mx-auto px-6 py-20">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {STATS.map((stat) => (
            <div key={stat.label} className="text-center">
              <AnimatedCounter target={stat.value} suffix={stat.suffix} />
              <p className="text-sm text-fg-muted mt-2">{stat.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <div className="text-center mb-14">
          <h2 className="text-3xl sm:text-4xl font-bold text-fg mb-4">
            Six-Layer Forensic Architecture
          </h2>
          <p className="text-fg-muted max-w-xl mx-auto">
            Every layer is designed for auditability, fairness, and legal admissibility.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map((feature) => (
            <div
              key={feature.title}
              className="group glass-card p-6 hover:scale-[1.02] transition-all duration-300"
            >
              <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 ${feature.iconColor} group-hover:scale-110 transition-transform`}>
                {feature.icon}
              </div>
              <h3 className="text-base font-semibold text-fg mb-2">{feature.title}</h3>
              <p className="text-sm text-fg-muted leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="max-w-6xl mx-auto px-6 py-20">
        <div className="text-center mb-14">
          <h2 className="text-3xl sm:text-4xl font-bold text-fg mb-4">
            How It Works
          </h2>
          <p className="text-fg-muted max-w-xl mx-auto">
            From recognition to compliance in a fully auditable pipeline.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            {
              step: "01",
              title: "Recognize & Record",
              desc: "Every inference creates a ForensicEvent — hash-chained and Merkle-verified automatically.",
              color: "text-cyan",
            },
            {
              step: "02",
              title: "Explain & Audit",
              desc: "LIME/SHAP explanations and bias audits are generated alongside every decision.",
              color: "text-amethyst",
            },
            {
              step: "03",
              title: "Prove & Comply",
              desc: "Zero-knowledge proofs, DID credentials, and OSCAL exports for regulatory submission.",
              color: "text-teal",
            },
          ].map((item) => (
            <div key={item.step} className="relative pl-8 border-l-2 border-border-theme">
              <span className={`absolute -left-3 top-0 w-6 h-6 rounded-full bg-page border-2 border-border-theme flex items-center justify-center text-[10px] font-bold ${item.color}`}>
                {item.step}
              </span>
              <h3 className="text-lg font-semibold text-fg mb-2">{item.title}</h3>
              <p className="text-sm text-fg-muted leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-6xl mx-auto px-6 py-20">
        <div className="relative p-12 rounded-2xl overflow-hidden border-sovereign">
          <div className="absolute top-0 right-0 w-64 h-64 bg-cyan/10 rounded-full blur-[80px] pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-amethyst/10 rounded-full blur-[80px] pointer-events-none" />

          <div className="relative text-center">
            <h2 className="text-3xl sm:text-4xl font-bold text-fg mb-4">Ready to get started?</h2>
            <p className="text-fg-muted mb-8 max-w-xl mx-auto">
              Deploy forensic-grade AI recognition with full compliance in minutes.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <button
                onClick={() => navigate("/signup")}
                className="btn-primary px-8 py-3 text-base"
              >
                Create Your Account
              </button>
              <button
                onClick={() => navigate("/pricing")}
                className="btn-ghost px-8 py-3 text-base"
              >
                Compare Plans
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border-theme py-10">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <img src="/logo.png" alt="FriendlyFace" className="w-6 h-6 opacity-50" />
              <span className="text-sm text-fg-faint">FriendlyFace v0.1.0</span>
            </div>
            <div className="flex items-center gap-6">
              <button onClick={() => navigate("/pricing")} className="text-sm text-fg-faint hover:text-fg-muted transition-colors">Pricing</button>
              <button onClick={() => navigate("/login")} className="text-sm text-fg-faint hover:text-fg-muted transition-colors">Sign In</button>
            </div>
            <span className="text-xs text-fg-faint">Forensic-Friendly AI Recognition</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
