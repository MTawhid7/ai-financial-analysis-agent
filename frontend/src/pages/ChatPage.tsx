import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useAuth } from "../hooks/useAuth";
import { ChatInterface } from "../components/chat/ChatInterface";
import { ConversationList } from "../components/sidebar/ConversationList";
import { MemoryPanel } from "../components/sidebar/MemoryPanel";

export function ChatPage() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [activeConvId, setActiveConvId] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleSignOut = async () => {
    await signOut();
    navigate({ to: "/" });
  };

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden">
      {/* ── Sidebar — collapsible with smooth transition ── */}
      <aside
        className={[
          "flex-shrink-0 flex flex-col border-r border-zinc-800/60 bg-zinc-900/80 backdrop-blur-sm",
          "transition-all duration-200 ease-in-out overflow-hidden",
          sidebarOpen ? "w-64" : "w-12",
        ].join(" ")}
      >
        {/* Header row */}
        <div className="flex items-center px-3 py-3 border-b border-zinc-800/60 gap-2 min-h-[52px]">
          {/* App icon (always visible) */}
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

        {/* Conversation list (only when expanded) */}
        {sidebarOpen && (
          <>
            <div className="flex-1 overflow-y-auto px-2 py-3 min-h-0">
              <ConversationList activeId={activeConvId} onSelect={setActiveConvId} />
            </div>

            <div className="px-2 py-2 border-t border-zinc-800/60">
              <MemoryPanel />
            </div>

            {/* User profile */}
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
                <div className="w-6 h-6 rounded-full bg-violet-700 flex items-center justify-center text-xs font-medium ring-1 ring-violet-600">
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

        {/* Icon rail (collapsed state) */}
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

      {/* ── Main chat area ── */}
      <main className="flex-1 flex flex-col min-w-0 bg-zinc-950">
        {activeConvId ? (
          <ChatInterface key={activeConvId} conversationId={activeConvId} />
        ) : (
          /* No active conversation — full-page landing */
          <div className="flex flex-col items-center justify-center h-full gap-8 px-4 text-center">
            <div className="w-16 h-16 rounded-2xl bg-violet-600/10 border border-violet-500/20 flex items-center justify-center mb-2">
              <svg className="w-8 h-8 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
            </div>
            <div className="space-y-3">
              <h1 className="text-3xl font-semibold text-zinc-100 tracking-tight">
                AI Financial Analyst
              </h1>
              <p className="text-zinc-500 text-sm max-w-sm leading-relaxed">
                Select a conversation from the sidebar or start a new one.
              </p>
            </div>
            <button
              onClick={() => {
                // Trigger new conversation via sidebar
                setSidebarOpen(true);
              }}
              className="px-5 py-2.5 rounded-full bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium transition-colors"
            >
              New conversation
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
