import { type KeyboardEvent, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useAuth } from "../hooks/useAuth";
import { ChatInterface } from "../components/chat/ChatInterface";
import { ConversationList } from "../components/sidebar/ConversationList";
import { MemoryPanel } from "../components/sidebar/MemoryPanel";
import { createConversation } from "../lib/api";

const LANDING_CHIPS = [
  "Analyse AAPL",
  "Compare AAPL vs MSFT",
  "What is a P/E ratio?",
  "What did we find about NVDA?",
];

export function ChatPage() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [activeConvId, setActiveConvId] = useState<string>("");
  const [initialMessage, setInitialMessage] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [landingText, setLandingText] = useState("");
  const [landingBusy, setLandingBusy] = useState(false);
  const landingRef = useRef<HTMLTextAreaElement>(null);

  const handleSignOut = async () => {
    await signOut();
    navigate({ to: "/" });
  };

  const handleLandingSend = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || landingBusy) return;
    setLandingBusy(true);
    try {
      const conv = await createConversation("New conversation");
      setInitialMessage(trimmed);
      setActiveConvId(conv.id);
      setLandingText("");
    } catch {
      // ignore — user can retry
    } finally {
      setLandingBusy(false);
    }
  };

  const handleLandingKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleLandingSend(landingText);
    }
  };

  const handleLandingInput = () => {
    const el = landingRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  const landingCanSend = !!landingText.trim() && !landingBusy;

  return (
    <div className="flex h-screen bg-[#0d0d0d] text-zinc-100 overflow-hidden">
      {/* ── Sidebar ── */}
      <aside
        className={[
          "flex-shrink-0 flex flex-col border-r border-zinc-800/60 bg-[#171717]",
          "transition-all duration-200 ease-in-out overflow-hidden",
          sidebarOpen ? "w-64" : "w-12",
        ].join(" ")}
      >
        {/* Header row */}
        <div className="flex items-center px-3 py-3 border-b border-zinc-800/60 gap-2 min-h-[52px]">
          <div className="flex-shrink-0 w-6 h-6 rounded-lg bg-violet-600/20 border border-violet-500/30 flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
          </div>

          {sidebarOpen && (
            <span className="flex-1 text-[13px] font-semibold text-zinc-200 tracking-tight truncate">
              Financial Analyst
            </span>
          )}

          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="flex-shrink-0 text-zinc-600 hover:text-zinc-400 transition-colors"
            aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            <svg className={`w-4 h-4 transition-transform duration-200 ${sidebarOpen ? "" : "rotate-180"}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            </svg>
          </button>
        </div>

        {sidebarOpen && (
          <>
            <div className="flex-1 overflow-y-auto px-2 py-3 min-h-0">
              <ConversationList activeId={activeConvId} onSelect={setActiveConvId} />
            </div>

            <div className="px-2 py-2 border-t border-zinc-800/60">
              <MemoryPanel />
            </div>

            <div className="flex items-center gap-2 px-3 py-3 border-t border-zinc-800/60">
              {user?.picture_url ? (
                <img
                  src={user.picture_url}
                  alt=""
                  className="w-6 h-6 rounded-full ring-1 ring-zinc-700"
                  onError={(e) => { e.currentTarget.style.display = "none"; }}
                  referrerPolicy="no-referrer"
                />
              ) : (
                <div className="w-6 h-6 rounded-full bg-zinc-700 flex items-center justify-center text-xs font-medium ring-1 ring-zinc-600">
                  {user?.display_name?.[0]?.toUpperCase() ?? "U"}
                </div>
              )}
              <span className="flex-1 text-xs text-zinc-500 truncate">
                {user?.display_name || user?.email}
              </span>
              <button
                onClick={handleSignOut}
                title="Sign out"
                className="text-zinc-600 hover:text-zinc-400 transition-colors"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
              </button>
            </div>
          </>
        )}

        {!sidebarOpen && (
          <div className="flex flex-col items-center gap-4 py-4">
            <button
              onClick={() => setActiveConvId("")}
              title="Conversations"
              className="text-zinc-600 hover:text-zinc-400 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </button>
          </div>
        )}
      </aside>

      {/* ── Main area ── */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#0d0d0d]">
        {activeConvId ? (
          <ChatInterface
            key={activeConvId}
            conversationId={activeConvId}
            initialMessage={initialMessage}
            onInitialMessageConsumed={() => setInitialMessage("")}
          />
        ) : (
          /* Gemini-style landing — centered input */
          <div className="flex flex-col items-center justify-center h-full px-6">
            <div className="w-full max-w-2xl flex flex-col items-center gap-8">
              {/* Icon */}
              <div className="w-14 h-14 rounded-2xl bg-zinc-800 border border-zinc-700 flex items-center justify-center">
                <svg className="w-7 h-7 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
              </div>

              {/* Heading */}
              <div className="text-center space-y-2">
                <h1 className="text-3xl font-semibold text-white tracking-tight">
                  What would you like to analyse?
                </h1>
                <p className="text-zinc-500 text-sm max-w-sm leading-relaxed">
                  Ask about stocks, compare companies, upload financials, or recall past research.
                </p>
              </div>

              {/* Suggestion chips */}
              <div className="flex flex-wrap gap-2 justify-center">
                {LANDING_CHIPS.map((chip) => (
                  <button
                    key={chip}
                    onClick={() => handleLandingSend(chip)}
                    disabled={landingBusy}
                    className="text-sm px-4 py-2 rounded-full border border-zinc-700 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200 transition-all duration-150 disabled:opacity-40"
                  >
                    {chip}
                  </button>
                ))}
              </div>

              {/* Gemini-style rounded input */}
              <div className={[
                "w-full rounded-2xl border bg-zinc-900 px-4 py-3",
                "transition-all duration-150",
                landingBusy
                  ? "border-zinc-800 opacity-60"
                  : "border-zinc-700 focus-within:border-zinc-500",
              ].join(" ")}>
                <textarea
                  ref={landingRef}
                  value={landingText}
                  onChange={(e) => setLandingText(e.target.value)}
                  onKeyDown={handleLandingKeyDown}
                  onInput={handleLandingInput}
                  placeholder="Ask about stocks, request analysis, compare companies…"
                  disabled={landingBusy}
                  rows={1}
                  className="w-full resize-none bg-transparent text-[14px] text-zinc-200 placeholder-zinc-600 outline-none leading-relaxed"
                  style={{ maxHeight: 160 }}
                />
                <div className="flex justify-end mt-2">
                  <button
                    onClick={() => handleLandingSend(landingText)}
                    disabled={!landingCanSend}
                    aria-label="Send"
                    className={[
                      "w-8 h-8 rounded-full flex items-center justify-center transition-all duration-150",
                      landingCanSend
                        ? "bg-white text-zinc-900 hover:bg-zinc-100 shadow-sm"
                        : "bg-zinc-800 text-zinc-600 cursor-not-allowed",
                    ].join(" ")}
                  >
                    <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                    </svg>
                  </button>
                </div>
              </div>

              <p className="text-[11px] text-zinc-700">
                Enter to send · Shift+Enter for new line
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
