/**
 * Centralized API client for FriendlyFace dashboard.
 *
 * - Uses VITE_API_URL env var if set, otherwise falls back to relative path.
 * - Provides typed fetch wrappers with error handling.
 * - Detects when backend is unreachable and enables demo mode.
 */

const API_BASE = import.meta.env.VITE_API_URL || "";

export interface ApiError {
    status: number;
    message: string;
    detail?: string;
}

export class ApiClientError extends Error {
    status: number;
    detail?: string;

    constructor(status: number, message: string, detail?: string) {
        super(message);
        this.name = "ApiClientError";
        this.status = status;
        this.detail = detail;
    }
}

let _demoMode = false;
let _demoModeChecked = false;

export function isDemoMode(): boolean {
    return _demoMode;
}

export function setDemoMode(value: boolean): void {
    _demoMode = value;
    _demoModeChecked = true;
}

/**
 * Check if the backend API is reachable. If not, enable demo mode.
 */
export async function checkBackendHealth(): Promise<boolean> {
    if (_demoModeChecked) return !_demoMode;

    try {
        const response = await fetch(`${API_BASE}/api/v1/health`, {
            signal: AbortSignal.timeout(5000),
        });
        if (!response.ok) {
            _demoMode = true;
        } else {
            // Verify the response is actually JSON (not an SPA HTML fallback)
            const contentType = response.headers.get("content-type") || "";
            if (!contentType.includes("application/json")) {
                _demoMode = true;
            } else {
                const data = await response.json();
                _demoMode = data.status !== "healthy";
            }
        }
    } catch {
        _demoMode = true;
    }
    _demoModeChecked = true;
    return !_demoMode;
}

/**
 * Demo-aware fetch: returns demo data for known endpoints when in demo mode,
 * otherwise performs a real fetch. Used by pages that call fetch() directly.
 */
export async function demoFetch(path: string, init?: RequestInit): Promise<Response> {
    if (_demoMode) {
        const demoData = await getDemoDataForPath(path);
        if (demoData !== undefined) {
            return new Response(JSON.stringify(demoData), {
                status: 200,
                headers: { "Content-Type": "application/json" },
            });
        }
    }
    return fetch(`${API_BASE}${path}`, init);
}

async function getDemoDataForPath(path: string): Promise<unknown | undefined> {
    const { demoDashboard, demoFLRounds, demoModels, demoGalleryCount,
            demoCompliance, demoEvents, demoBundles, demoFairnessStatus,
            demoFairnessAudits, demoMerkleRoot, demoHealth } = await import("./demoData");
    const p = path.split("?")[0]; // strip query params
    const map: Record<string, unknown> = {
        "/api/v1/dashboard": demoDashboard,
        "/api/v1/health": demoHealth,
        "/api/v1/fl/rounds": demoFLRounds,
        "/api/v1/recognition/models": demoModels,
        "/api/v1/gallery/count": demoGalleryCount,
        "/api/v1/governance/compliance": demoCompliance,
        "/api/v1/events": demoEvents,
        "/api/v1/bundles": demoBundles,
        "/api/v1/fairness/status": demoFairnessStatus,
        "/api/v1/fairness/audits": demoFairnessAudits,
        "/api/v1/merkle/root": demoMerkleRoot,
    };
    return map[p];
}

/**
 * Generic GET request.
 */
export async function apiGet<T>(path: string): Promise<T> {
    const url = `${API_BASE}${path}`;
    const response = await fetch(url);
    if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new ApiClientError(
            response.status,
            body.message || `Request failed: ${response.status}`,
            body.detail
        );
    }
    return response.json();
}

/**
 * Generic POST request.
 */
export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
    const url = `${API_BASE}${path}`;
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new ApiClientError(
            response.status,
            data.message || `Request failed: ${response.status}`,
            data.detail
        );
    }
    return response.json();
}

/**
 * Generic DELETE request.
 */
export async function apiDelete<T>(path: string): Promise<T> {
    const url = `${API_BASE}${path}`;
    const response = await fetch(url, { method: "DELETE" });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new ApiClientError(
            response.status,
            data.message || `Request failed: ${response.status}`,
            data.detail
        );
    }
    return response.json();
}
