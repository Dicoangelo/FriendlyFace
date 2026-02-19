import { useNavigate } from "react-router-dom";

interface PlanTier {
  name: string;
  price: string;
  period: string;
  description: string;
  features: string[];
  cta: string;
  highlighted?: boolean;
  plan: string;
}

const PLANS: PlanTier[] = [
  {
    name: "Starter",
    price: "$49",
    period: "/month",
    description: "For teams getting started with forensic AI recognition.",
    plan: "starter",
    cta: "Get Started",
    features: [
      "Forensic event chain",
      "Merkle verification",
      "Basic recognition (PCA+SVM)",
      "100 API calls/min",
      "Email support",
      "1 API key",
    ],
  },
  {
    name: "Professional",
    price: "$199",
    period: "/month",
    description: "Full forensic capabilities for professional teams.",
    plan: "professional",
    cta: "Upgrade to Pro",
    highlighted: true,
    features: [
      "Everything in Starter",
      "Model training (recognition/train)",
      "Federated Learning (FL)",
      "DID/VC issuance",
      "500 API calls/min",
      "10 API keys",
      "Priority support",
    ],
  },
  {
    name: "Enterprise",
    price: "$499",
    period: "/month",
    description: "Maximum security and compliance for enterprise deployment.",
    plan: "enterprise",
    cta: "Contact Sales",
    features: [
      "Everything in Professional",
      "Zero-knowledge proofs (ZK)",
      "OSCAL compliance export",
      "2,000 API calls/min",
      "Unlimited API keys",
      "SSO / OIDC integration",
      "Dedicated support",
    ],
  },
];

export default function PricingPage() {
  const navigate = useNavigate();
  const token = localStorage.getItem("ff_token");

  const handleSelect = async (plan: string) => {
    if (!token) {
      navigate("/signup");
      return;
    }

    if (plan === "starter") {
      navigate("/");
      return;
    }

    try {
      const res = await fetch("/api/v1/billing/checkout", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          plan,
          success_url: window.location.origin + "/pricing?success=true",
          cancel_url: window.location.origin + "/pricing?canceled=true",
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        alert(data.detail || "Failed to create checkout session");
        return;
      }

      const data = await res.json();
      if (data.checkout_url) {
        window.location.href = data.checkout_url;
      }
    } catch {
      alert("Failed to connect to billing service");
    }
  };

  return (
    <div className="min-h-screen bg-page">
      {/* Nav */}
      <nav className="border-b border-border-theme bg-sidebar/50 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/landing")}>
            <img src="/logo.png" alt="FriendlyFace" className="w-8 h-8" />
            <span className="text-xl font-bold gradient-text">FriendlyFace</span>
          </div>
          <div className="flex items-center gap-4">
            {token ? (
              <button
                onClick={() => navigate("/")}
                className="text-sm text-fg-muted hover:text-fg transition-colors"
              >
                Dashboard
              </button>
            ) : (
              <button
                onClick={() => navigate("/login")}
                className="text-sm text-fg-muted hover:text-fg transition-colors"
              >
                Sign In
              </button>
            )}
          </div>
        </div>
      </nav>

      {/* Pricing header */}
      <div className="max-w-6xl mx-auto px-6 py-16 text-center">
        <h1 className="text-4xl font-bold text-fg mb-4">Simple, transparent pricing</h1>
        <p className="text-lg text-fg-muted max-w-xl mx-auto">
          Choose the plan that fits your forensic AI needs. All plans include hash-chained audit trails.
        </p>
      </div>

      {/* Plans */}
      <div className="max-w-6xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PLANS.map((plan) => (
            <div
              key={plan.name}
              className={`relative p-8 rounded-xl border ${
                plan.highlighted
                  ? "border-cyan bg-cyan/5 shadow-lg shadow-cyan/10"
                  : "border-border-theme bg-sidebar/50"
              }`}
            >
              {plan.highlighted && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-cyan text-white text-xs font-medium rounded-full">
                  Most Popular
                </div>
              )}
              <h3 className="text-xl font-semibold text-fg">{plan.name}</h3>
              <div className="mt-4 flex items-baseline gap-1">
                <span className="text-4xl font-bold text-fg">{plan.price}</span>
                <span className="text-fg-muted">{plan.period}</span>
              </div>
              <p className="mt-3 text-sm text-fg-muted">{plan.description}</p>

              <button
                onClick={() => handleSelect(plan.plan)}
                className={`mt-6 w-full py-2.5 rounded-lg font-medium transition-colors ${
                  plan.highlighted
                    ? "bg-cyan text-white hover:bg-cyan/90"
                    : "border border-border-theme text-fg hover:bg-fg/5"
                }`}
              >
                {plan.cta}
              </button>

              <ul className="mt-8 space-y-3">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-3 text-sm text-fg-muted">
                    <svg
                      className="w-5 h-5 text-teal flex-shrink-0 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
