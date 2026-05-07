// In development, Vite proxies /api/* to FastAPI.
// In production, set VITE_API_BASE to the full FastAPI URL.
export const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID ?? "";
