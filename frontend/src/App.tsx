import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import { useState } from "react";
import { useTheme } from "./components/ThemeProvider";
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

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "grid" },
  { to: "/events/live", label: "Live Events", icon: "zap" },
  { to: "/events", label: "Events Table", icon: "list" },
  { to: "/bundles", label: "Bundles", icon: "package" },
  { to: "/did", label: "DID / VC", icon: "key" },
  { to: "/zk", label: "ZK Proofs", icon: "shield" },
  { to: "/fl", label: "FL Simulations", icon: "cpu" },
  { to: "/bias", label: "Bias Audits", icon: "scale" },
  { to: "/explainability", label: "Explainability", icon: "search" },
  { to: "/consent", label: "Consent", icon: "check" },
];

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
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-page">
        {/* Sidebar */}
        <aside
          className={`${
            sidebarOpen ? "w-56" : "w-14"
          } bg-sidebar border-r border-border-theme flex flex-col transition-all duration-200 flex-shrink-0`}
        >
          <div className="flex items-center justify-between px-3 py-4 border-b border-border-theme">
            {sidebarOpen && (
              <div className="flex items-center gap-2">
                <img
                  src="https://www.gravatar.com/avatar/868b54ba79b7e4eb97bc0f8bac8ed47f?s=200&d=identicon"
                  alt="FriendlyFace"
                  className="w-7 h-7 rounded-lg"
                />
                <span className="text-lg font-bold tracking-tight gradient-text">FriendlyFace</span>
              </div>
            )}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-fg-faint hover:text-fg-secondary p-1 transition-colors"
              title={sidebarOpen ? "Collapse" : "Expand"}
            >
              {sidebarOpen ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" /></svg>
              ) : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" /></svg>
              )}
            </button>
          </div>
          <nav className="flex-1 py-2 overflow-y-auto">
            {NAV_ITEMS.map((item) => (
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
          <header className="bg-sidebar/50 backdrop-blur-sm border-b border-border-theme px-6 py-3 flex items-center justify-between flex-shrink-0">
            <h1 className="text-lg font-semibold text-fg">FriendlyFace Dashboard</h1>
            <div className="flex items-center gap-3">
              <ThemeToggle />
              <HealthBadge />
            </div>
          </header>

          {/* Page content */}
          <main className="flex-1 overflow-auto p-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/events/live" element={<EventStream />} />
              <Route path="/events" element={<EventsTable />} />
              <Route path="/bundles" element={<Bundles />} />
              <Route path="/did" element={<DIDManagement />} />
              <Route path="/zk" element={<ZKProofs />} />
              <Route path="/fl" element={<FLSimulations />} />
              <Route path="/bias" element={<BiasAudits />} />
              <Route path="/explainability" element={<Explainability />} />
              <Route path="/consent" element={<ConsentManagement />} />
              <Route path="/index.html" element={<Navigate to="/" replace />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

function HealthBadge() {
  const [status, setStatus] = useState<"ok" | "error" | "loading">("loading");

  useState(() => {
    fetch("/api/v1/health")
      .then((r) => (r.ok ? setStatus("ok") : setStatus("error")))
      .catch(() => setStatus("error"));
  });

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
