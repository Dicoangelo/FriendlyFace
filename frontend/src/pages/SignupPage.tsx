import { useState, FormEvent } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function SignupPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch("/api/v1/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, name: name || undefined }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Registration failed");
      }

      const data = await res.json();
      localStorage.setItem("ff_token", data.token);
      localStorage.setItem("ff_user_id", data.user_id);
      localStorage.setItem("ff_email", data.email);
      navigate("/");
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-page flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <img src="/logo.png" alt="FriendlyFace" className="w-12 h-12 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-fg">Create your account</h1>
          <p className="text-sm text-fg-muted mt-2">
            Start with a free Starter plan
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="p-8 rounded-xl border border-border-theme bg-sidebar/50 space-y-5"
        >
          {error && (
            <div className="p-3 rounded-lg bg-rose-ember/10 border border-rose-ember/20 text-rose-ember text-sm">
              {error}
            </div>
          )}

          <div>
            <label htmlFor="name" className="block text-sm font-medium text-fg mb-1.5">
              Name (optional)
            </label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg border border-border-theme bg-page text-fg placeholder-fg-faint focus:outline-none focus:border-cyan focus:ring-1 focus:ring-cyan"
              placeholder="Your name"
            />
          </div>

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-fg mb-1.5">
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg border border-border-theme bg-page text-fg placeholder-fg-faint focus:outline-none focus:border-cyan focus:ring-1 focus:ring-cyan"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-fg mb-1.5">
              Password
            </label>
            <input
              id="password"
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg border border-border-theme bg-page text-fg placeholder-fg-faint focus:outline-none focus:border-cyan focus:ring-1 focus:ring-cyan"
              placeholder="Min 8 characters"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-cyan text-white font-medium hover:bg-cyan/90 disabled:opacity-50 transition-colors"
          >
            {loading ? "Creating account..." : "Create Account"}
          </button>

          <p className="text-center text-sm text-fg-muted">
            Already have an account?{" "}
            <Link to="/login" className="text-cyan hover:text-cyan-dim">
              Sign in
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}
