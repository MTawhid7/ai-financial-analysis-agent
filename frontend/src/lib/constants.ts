// In development, Vite proxies /api/* to FastAPI.
// In production, set VITE_API_BASE to the full FastAPI URL.
// Use VITE_API_BASE if explicitly set to a non-empty value; otherwise default to /api
// (which Vite proxies to FastAPI at localhost:8000 during development).
export const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID ?? "";
