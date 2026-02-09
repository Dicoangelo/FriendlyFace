import { BrowserRouter, Routes, Route, NavLink, Navigate, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";
import { useTheme } from "./hooks/useTheme";
import Dashboard from "./pages/Dashboard";
import EventStream from "./pages/EventStream";
import EventsTable from "./pages/EventsTable";
import Bundles from "./pages/Bundles";
import DIDManagement from "./pages/DIDManagement";
import ZKProofs from "./pages/ZKProofs";
import FLSimulations from "./pages/FLSimulations";
import BiasAudits from "./pages/BiasAudits";
import ConsentManagement from "./pages/ConsentManagement";
import Explainability from "./pages/Explainability";
import Recognition from "./pages/Recognition";
import Compliance from "./pages/Compliance";
import DataErasure from "./pages/DataErasure";
import RetentionPolicies from "./pages/RetentionPolicies";
import AdminOps from "./pages/AdminOps";
import MerkleExplorer from "./pages/MerkleExplorer";
import ProvenanceExplorer from "./pages/ProvenanceExplorer";

interface NavItem {
  to: string;
  label: string;
  icon: string;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    title: "Overview",
    items: [
      { to: "/", label: "Dashboard", icon: "grid" },
    ],
  },
  {
    title: "Forensic",
    items: [
      { to: "/events/live", label: "Live Events", icon: "zap" },
      { to: "/events", label: "Events Table", icon: "list" },
      { to: "/bundles", label: "Bundles", icon: "package" },
      { to: "/merkle", label: "Merkle Tree", icon: "hash" },
      { to: "/provenance", label: "Provenance", icon: "gitbranch" },
    ],
  },
  {
    title: "AI / ML",
    items: [
      { to: "/recognition", label: "Recognition", icon: "eye" },
      { to: "/fl", label: "FL Simulations", icon: "cpu" },
      { to: "/explainability", label: "Explainability", icon: "search" },
      { to: "/bias", label: "Bias Audits", icon: "scale" },
    ],
  },
  {
    title: "Governance",
    items: [
      { to: "/compliance", label: "Compliance", icon: "clipboard" },
      { to: "/consent", label: "Consent", icon: "check" },
      { to: "/erasure", label: "Data Erasure", icon: "trash" },
      { to: "/retention", label: "Retention", icon: "clock" },
      { to: "/did", label: "DID / VC", icon: "key" },
      { to: "/zk", label: "ZK Proofs", icon: "shield" },
    ],
  },
  {
    title: "Admin",
    items: [
      { to: "/admin", label: "Operations", icon: "settings" },
    ],
  },
];

const PAGE_TITLES: Record<string, string> = {
  "/": "Dashboard",
  "/events/live": "Live Event Stream",
  "/events": "Forensic Events",
  "/bundles": "Forensic Bundles",
  "/recognition": "Face Recognition",
  "/fl": "Federated Learning",
  "/explainability": "Explainability",
  "/bias": "Bias Audits",
  "/compliance": "EU AI Act Compliance",
  "/consent": "Consent Management",
  "/erasure": "Data Erasure",
  "/retention": "Retention Policies",
  "/did": "DID / Verifiable Credentials",
  "/zk": "ZK Proofs",
  "/admin": "Admin Operations",
  "/merkle": "Merkle Tree",
  "/provenance": "Provenance Explorer",
};

const PAGE_DESCRIPTIONS: Record<string, string> = {
  "/": "System health, model status, and forensic chain overview",
  "/events/live": "Real-time Server-Sent Events from the forensic pipeline",
  "/events": "Browse, filter, and inspect all forensic events",
  "/bundles": "Cryptographically signed evidence bundles with Merkle verification",
  "/recognition": "PCA + SVM facial recognition with voice biometrics",
  "/fl": "Federated learning simulations with poisoning detection and DP-FedAvg",
  "/explainability": "LIME, SHAP, and SDD saliency maps for model interpretability",
  "/bias": "Demographic parity and equalized odds audits across groups",
  "/compliance": "EU AI Act compliance reports and OSCAL export",
  "/consent": "Grant, check, and revoke subject consent with full audit trail",
  "/erasure": "GDPR Article 17 — cryptographic erasure of subject data",
  "/retention": "Automated data lifecycle management with retention policies",
  "/did": "Decentralized Identifiers and W3C Verifiable Credentials",
  "/zk": "Schnorr zero-knowledge proofs for privacy-preserving verification",
  "/admin": "Database backups, migrations, and system operations",
  "/merkle": "Append-only Merkle tree root and cryptographic inclusion proofs",
  "/provenance": "Trace artifact lineage: dataset → training → model → inference → explanation",
};

const NAV_ICONS: Record<string, React.ReactNode> = {
  grid: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" /></svg>,
  zap: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
  list: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" /></svg>,
  package: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" /></svg>,
  key: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" /></svg>,
  shield: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>,
  cpu: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m-2 6h2m14-6h2m-2 6h2M7 7h10v10H7V7z" /></svg>,
  scale: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" /></svg>,
  search: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>,
  check: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  eye: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>,
  clipboard: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" /></svg>,
  trash: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>,
  clock: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  settings: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
  hash: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" /></svg>,
  gitbranch: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 3v12m0 0a3 3 0 103 3 3 3 0 00-3-3zm12-6a3 3 0 10-3-3 3 3 0 003 3zm0 0v3a3 3 0 01-3 3H9" /></svg>,
};

function NotFound() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-fg-faint">404</h1>
        <p className="mt-4 text-xl text-fg-muted">Page not found</p>
        <a href="/" className="mt-4 inline-block text-cyan hover:text-cyan-dim">
          Back to Dashboard
        </a>
      </div>
    </div>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      className="p-2 rounded-lg text-fg-muted hover:text-fg hover:bg-fg/5 transition-colors"
      title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
    >
      {theme === "dark" ? (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
      ) : (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
        </svg>
      )}
    </button>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}

function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();
  const pageTitle = PAGE_TITLES[location.pathname] || "FriendlyFace";
  const pageDescription = PAGE_DESCRIPTIONS[location.pathname];

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  return (
      <div className="flex h-screen bg-page grid-bg">
        {/* Mobile backdrop */}
        {mobileMenuOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-30 md:hidden"
            onClick={() => setMobileMenuOpen(false)}
          />
        )}

        {/* Sidebar */}
        <aside
          className={`${
            sidebarOpen ? "w-56" : "w-14"
          } bg-sidebar border-r border-border-theme flex flex-col transition-all duration-200 flex-shrink-0
          ${mobileMenuOpen ? "fixed inset-y-0 left-0 z-40 w-56 shadow-2xl animate-slide-in-left" : "hidden md:flex"}
          `}
        >
          <div className="flex items-center justify-between px-3 py-4 border-b border-border-theme">
            {(sidebarOpen || mobileMenuOpen) && (
              <div className="flex items-center gap-2">
                <img
                  src="/logo.png"
                  alt="FriendlyFace"
                  className="w-7 h-7"
                />
                <span className="text-lg font-bold tracking-tight gradient-text">FriendlyFace</span>
              </div>
            )}
            <button
              onClick={() => {
                if (mobileMenuOpen) setMobileMenuOpen(false);
                else setSidebarOpen(!sidebarOpen);
              }}
              className="text-fg-faint hover:text-fg-secondary p-1 transition-colors"
              title={sidebarOpen ? "Collapse" : "Expand"}
              aria-label="Toggle sidebar"
            >
              {sidebarOpen || mobileMenuOpen ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" /></svg>
              ) : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" /></svg>
              )}
            </button>
          </div>
          <nav className="flex-1 py-2 overflow-y-auto">
            {NAV_SECTIONS.map((section) => (
              <div key={section.title} className="mb-1">
                {sidebarOpen && (
                  <p className="px-4 pt-3 pb-1 text-[10px] font-semibold uppercase tracking-wider text-fg-faint">
                    {section.title}
                  </p>
                )}
                {!sidebarOpen && section.title !== "Overview" && (
                  <div className="mx-3 my-1 border-t border-border-theme" />
                )}
                {section.items.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.to === "/"}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-3 py-2 mx-1 rounded-lg text-sm transition-all ${
                        isActive
                          ? "bg-cyan/10 text-cyan border border-cyan/20"
                          : "text-fg-muted hover:bg-fg/5 hover:text-fg border border-transparent"
                      }`
                    }
                  >
                    <span className="flex-shrink-0">{NAV_ICONS[item.icon]}</span>
                    {sidebarOpen && <span>{item.label}</span>}
                  </NavLink>
                ))}
              </div>
            ))}
          </nav>
          {sidebarOpen && (
            <div className="px-3 py-3 border-t border-border-theme text-xs text-fg-faint">
              v0.1.0 — Forensic Layer
            </div>
          )}
        </aside>

        {/* Main content */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Top bar */}
          <header className="bg-sidebar/50 backdrop-blur-sm border-b border-border-theme px-4 md:px-6 py-3 flex items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-3">
              {/* Mobile hamburger */}
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="md:hidden text-fg-muted hover:text-fg p-1 -ml-1"
                aria-label="Open menu"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              <div>
                <h1 className="text-lg font-semibold text-fg">{pageTitle}</h1>
              {pageDescription && (
                <p className="text-xs text-fg-faint mt-0.5 hidden sm:block">{pageDescription}</p>
              )}
              </div>
            </div>
            <div className="flex items-center gap-2 sm:gap-3">
              <ThemeToggle />
              <HealthBadge />
            </div>
          </header>

          {/* Page content */}
          <main className="flex-1 overflow-auto p-6 max-w-screen-2xl mx-auto w-full">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/events/live" element={<EventStream />} />
              <Route path="/events" element={<EventsTable />} />
              <Route path="/bundles" element={<Bundles />} />
              <Route path="/did" element={<DIDManagement />} />
              <Route path="/zk" element={<ZKProofs />} />
              <Route path="/recognition" element={<Recognition />} />
              <Route path="/fl" element={<FLSimulations />} />
              <Route path="/bias" element={<BiasAudits />} />
              <Route path="/explainability" element={<Explainability />} />
              <Route path="/compliance" element={<Compliance />} />
              <Route path="/consent" element={<ConsentManagement />} />
              <Route path="/erasure" element={<DataErasure />} />
              <Route path="/retention" element={<RetentionPolicies />} />
              <Route path="/admin" element={<AdminOps />} />
              <Route path="/merkle" element={<MerkleExplorer />} />
              <Route path="/provenance" element={<ProvenanceExplorer />} />
              <Route path="/index.html" element={<Navigate to="/" replace />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </main>

          {/* Footer status bar */}
          <footer className="bg-sidebar/50 backdrop-blur-sm border-t border-border-theme px-4 md:px-6 py-2 flex items-center justify-between text-xs text-fg-faint flex-shrink-0">
            <span className="truncate">FriendlyFace v0.1.0</span>
            <div className="flex items-center gap-3 sm:gap-4 flex-shrink-0">
              <span className="hidden sm:inline">Hash-chained</span>
              <span className="hidden sm:inline">Merkle-verified</span>
              <HealthBadge />
            </div>
          </footer>
        </div>
      </div>
  );
}

function HealthBadge() {
  const [status, setStatus] = useState<"ok" | "error" | "loading">("loading");

  useEffect(() => {
    fetch("/api/v1/health")
      .then((r) => (r.ok ? setStatus("ok") : setStatus("error")))
      .catch(() => setStatus("error"));
  }, []);

  const colors = {
    ok: "bg-teal/10 text-teal border border-teal/20",
    error: "bg-rose-ember/10 text-rose-ember border border-rose-ember/20",
    loading: "bg-fg/5 text-fg-muted border border-fg-faint/20",
  };

  return (
    <span className={`px-2 py-1 rounded-lg text-xs font-medium ${colors[status]}`}>
      {status === "ok" ? "API Connected" : status === "error" ? "API Offline" : "Checking..."}
    </span>
  );
}
