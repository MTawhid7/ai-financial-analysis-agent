/**
 * Typed fetch wrappers for all FastAPI endpoints.
 * All requests include credentials (httpOnly cookie).
 */

import { API_BASE } from "./constants";

export interface UserProfile {
  id: string;
  email: string;
  display_name: string;
  picture_url: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
}

export interface MessageOut {
  role: "user" | "assistant";
  content: string;
  intent: string;
  tickers: string;
  charts?: any[];
  report_id?: string;
  created_at: number;
}

export interface ConversationDetail {
  id: string;
  title: string;
  messages: MessageOut[];
}

export interface SummaryOut {
  id: number;
  tickers: string;
  summary_text: string;
  created_at: number;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export async function signInWithGoogle(idToken: string): Promise<UserProfile> {
  const res = await fetch(`${API_BASE}/auth/google`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id_token: idToken }),
  });
  if (!res.ok) throw new Error("Sign-in failed");
  return res.json();
}

export async function getMe(): Promise<UserProfile> {
  const res = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
  if (!res.ok) throw new Error("Not authenticated");
  return res.json();
}

export async function logout(): Promise<void> {
  await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" });
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

export async function listConversations(): Promise<ConversationSummary[]> {
  const res = await fetch(`${API_BASE}/conversations`, { credentials: "include" });
  if (!res.ok) throw new Error("Failed to load conversations");
  return res.json();
}

export async function createConversation(title: string = "New conversation"): Promise<ConversationSummary> {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("Failed to create conversation");
  return res.json();
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const res = await fetch(`${API_BASE}/conversations/${id}`, { credentials: "include" });
  if (!res.ok) throw new Error("Failed to load conversation");
  return res.json();
}

export async function updateConversationTitle(id: string, title: string): Promise<ConversationSummary> {
  const res = await fetch(`${API_BASE}/conversations/${id}`, {
    method: "PATCH",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("Failed to update title");
  return res.json();
}

export async function deleteConversation(id: string): Promise<void> {
  await fetch(`${API_BASE}/conversations/${id}`, {
    method: "DELETE",
    credentials: "include",
  });
}

// ---------------------------------------------------------------------------
// Chat (returns event_id for SSE stream)
// ---------------------------------------------------------------------------

export async function sendMessage(conversationId: string, message: string): Promise<{ event_id: string }> {
  const res = await fetch(`${API_BASE}/chat/${conversationId}`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) throw new Error("Failed to send message");
  return res.json();
}

export function createEventSource(eventId: string): EventSource {
  return new EventSource(`${API_BASE}/stream/${eventId}`, { withCredentials: true });
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

export async function getPreferences(): Promise<Record<string, string>> {
  const res = await fetch(`${API_BASE}/memory/preferences`, { credentials: "include" });
  if (!res.ok) throw new Error("Failed to load preferences");
  return res.json();
}

export async function getSummaries(limit = 20): Promise<SummaryOut[]> {
  const res = await fetch(`${API_BASE}/memory/summaries?limit=${limit}`, { credentials: "include" });
  if (!res.ok) throw new Error("Failed to load summaries");
  return res.json();
}

export async function clearAllMemory(): Promise<void> {
  await fetch(`${API_BASE}/memory/clear`, { method: "POST", credentials: "include" });
}

// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------

export async function uploadFile(file: File): Promise<Record<string, unknown>> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/files/upload`, {
    method: "POST",
    credentials: "include",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export interface ExportAvailability {
  pdf: boolean;
  docx: boolean;
  xlsx: boolean;
}

export async function getExportAvailable(): Promise<ExportAvailability> {
  const res = await fetch(`${API_BASE}/export/available`, { credentials: "include" });
  if (!res.ok) return { pdf: false, docx: true, xlsx: true };
  return res.json();
}

async function _exportBlob(url: string): Promise<Blob> {
  const res = await fetch(url, { method: "POST", credentials: "include" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Export failed");
  }
  return res.blob();
}

export const exportPdf  = (id: string) => _exportBlob(`${API_BASE}/export/pdf/${id}`);
export const exportDocx = (id: string) => _exportBlob(`${API_BASE}/export/docx/${id}`);
export const exportXlsx = (id: string) => _exportBlob(`${API_BASE}/export/xlsx/${id}`);

// ---------------------------------------------------------------------------
// Feedback
// ---------------------------------------------------------------------------

export async function submitFeedback(
  conversationId: string,
  messageIndex: number,
  rating: 1 | -1,
): Promise<void> {
  await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      message_index: messageIndex,
      rating,
    }),
  });
}

export async function getFeedbackStats(
  conversationId: string,
): Promise<Record<number, 1 | -1>> {
  const res = await fetch(`${API_BASE}/feedback/stats/${conversationId}`, {
    credentials: "include",
  });
  if (!res.ok) return {};
  return res.json();
}

// ---------------------------------------------------------------------------
// Sources (provenance — returns analysis citations for a report)
// ---------------------------------------------------------------------------

export interface WebSource {
  ticker: string;
  title: string;
  url: string;
  score: number;
}

export async function getReportSources(reportId: string): Promise<{
  tickers: string;
  analysis: Record<string, Record<string, { value: unknown; source_tool: string; observation_step: number }>>;
  web_sources: WebSource[];
}> {
  const res = await fetch(`${API_BASE}/reports/${reportId}/sources`, {
    credentials: "include",
  });
  if (!res.ok) throw new Error("Sources not found");
  return res.json();
}
