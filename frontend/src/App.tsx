import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import EventStream from "./pages/EventStream";
import EventsTable from "./pages/EventsTable";
import Bundles from "./pages/Bundles";
import DIDManagement from "./pages/DIDManagement";
import ZKProofs from "./pages/ZKProofs";
import FLSimulations from "./pages/FLSimulations";
import BiasAudits from "./pages/BiasAudits";
import ConsentManagement from "./pages/ConsentManagement";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "📊" },
  { to: "/events/live", label: "Live Events", icon: "⚡" },
  { to: "/events", label: "Events Table", icon: "📋" },
  { to: "/bundles", label: "Bundles", icon: "📦" },
  { to: "/did", label: "DID / VC", icon: "🔑" },
  { to: "/zk", label: "ZK Proofs", icon: "🛡️" },
  { to: "/fl", label: "FL Simulations", icon: "🤖" },
  { to: "/bias", label: "Bias Audits", icon: "⚖️" },
  { to: "/consent", label: "Consent", icon: "✅" },
];

function NotFound() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-gray-300">404</h1>
        <p className="mt-4 text-xl text-gray-500">Page not found</p>
        <a href="/" className="mt-4 inline-block text-blue-600 hover:underline">
          Back to Dashboard
        </a>
      </div>
    </div>
  );
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-50">
        {/* Sidebar */}
        <aside
          className={`${
            sidebarOpen ? "w-56" : "w-14"
          } bg-gray-900 text-white flex flex-col transition-all duration-200 flex-shrink-0`}
        >
          <div className="flex items-center justify-between px-3 py-4 border-b border-gray-700">
            {sidebarOpen && (
              <span className="text-lg font-bold tracking-tight">FriendlyFace</span>
            )}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-gray-400 hover:text-white p-1"
              title={sidebarOpen ? "Collapse" : "Expand"}
            >
              {sidebarOpen ? "◀" : "▶"}
            </button>
          </div>
          <nav className="flex-1 py-2 overflow-y-auto">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 mx-1 rounded text-sm ${
                    isActive
                      ? "bg-blue-600 text-white"
                      : "text-gray-300 hover:bg-gray-800 hover:text-white"
                  }`
                }
              >
                <span className="text-base">{item.icon}</span>
                {sidebarOpen && <span>{item.label}</span>}
              </NavLink>
            ))}
          </nav>
          {sidebarOpen && (
            <div className="px-3 py-3 border-t border-gray-700 text-xs text-gray-500">
              v0.1.0 — Forensic Layer
            </div>
          )}
        </aside>

        {/* Main content */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Top bar */}
          <header className="bg-white border-b px-6 py-3 flex items-center justify-between flex-shrink-0">
            <h1 className="text-lg font-semibold text-gray-800">FriendlyFace Dashboard</h1>
            <HealthBadge />
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
    ok: "bg-green-100 text-green-800",
    error: "bg-red-100 text-red-800",
    loading: "bg-gray-100 text-gray-500",
  };

  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${colors[status]}`}>
      {status === "ok" ? "API Connected" : status === "error" ? "API Offline" : "Checking..."}
    </span>
  );
}
