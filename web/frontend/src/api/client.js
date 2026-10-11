const API = import.meta.env?.VITE_API_URL ??
    "http://localhost:8000";
export class ApiError extends Error {
    constructor(status, detail) {
        super(detail);
        this.status = status;
    }
}
async function req(method, path, body) {
    const res = await fetch(API + path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (res.status === 204)
        return undefined;
    const data = await res.json().catch(() => ({}));
    if (!res.ok)
        throw new ApiError(res.status, data.detail ?? res.statusText);
    return data;
}
export const runsApi = {
    list: () => req("GET", "/api/runs"),
    get: (id) => req("GET", `/api/runs/${id}`),
    create: (b) => req("POST", "/api/runs", b),
    update: (id, b) => req("PUT", `/api/runs/${id}`, b),
    remove: (id) => req("DELETE", `/api/runs/${id}`),
    telemetry: (id, from, to, step) => req("GET", `/api/runs/${id}/telemetry?from=${from ?? 0}&to=${to ?? 1e9}&step=${step ?? 1}`),
    events: (id) => req("GET", `/api/runs/${id}/events`),
    report: (id) => req("GET", `/api/runs/${id}/report`),
};
export const routesApi = {
    list: () => req("GET", "/api/routes"),
    get: (id) => req("GET", `/api/routes/${id}`),
    create: (b) => req("POST", "/api/routes", b),
    update: (id, b) => req("PUT", `/api/routes/${id}`, b),
    remove: (id) => req("DELETE", `/api/routes/${id}`),
};
export const authApi = {
    login: (body) => req("POST", "/api/auth/login", body),
    register: (body) => req("POST", "/api/auth/register", body),
};
