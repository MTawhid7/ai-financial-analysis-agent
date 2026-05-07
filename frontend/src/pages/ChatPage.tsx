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
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      {/* Sidebar */}
      {sidebarOpen && (
        <aside className="w-64 flex-shrink-0 flex flex-col border-r border-zinc-800 bg-zinc-900">
          {/* Header */}
          <div className="flex items-center gap-2 px-4 py-3 border-b border-zinc-800">
            <span className="text-lg">📊</span>
            <span className="text-sm font-semibold text-zinc-200 flex-1">Financial Analyst</span>
            <button
              onClick={() => setSidebarOpen(false)}
              className="text-zinc-500 hover:text-zinc-300 transition-colors"
              aria-label="Close sidebar"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              </svg>
            </button>
          </div>

          {/* Conversation list */}
          <div className="flex-1 overflow-y-auto px-2 py-3">
            <ConversationList activeId={activeConvId} onSelect={setActiveConvId} />
          </div>

          {/* Memory panel */}
          <div className="px-2 py-2 border-t border-zinc-800">
            <MemoryPanel />
          </div>

          {/* User profile */}
          <div className="flex items-center gap-2 px-3 py-3 border-t border-zinc-800">
            {user?.picture_url ? (
              <img
                src={user.picture_url}
                alt=""
                className="w-6 h-6 rounded-full"
                onError={(e) => { e.currentTarget.style.display = "none"; }}
                referrerPolicy="no-referrer"
              />
            ) : (
              <div className="w-6 h-6 rounded-full bg-violet-600 flex items-center justify-center text-xs font-medium">
                {user?.display_name?.[0] ?? "U"}
              </div>
            )}
            <span className="flex-1 text-xs text-zinc-400 truncate">
              {user?.display_name || user?.email}
            </span>
            <button
              onClick={handleSignOut}
              className="text-zinc-500 hover:text-zinc-300 transition-colors"
              title="Sign out"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        </aside>
      )}

      {/* Main chat area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Topbar when sidebar is closed */}
        {!sidebarOpen && (
          <div className="flex items-center gap-2 px-4 py-2.5 border-b border-zinc-800">
            <button
              onClick={() => setSidebarOpen(true)}
              className="text-zinc-500 hover:text-zinc-300 transition-colors"
              aria-label="Open sidebar"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <span className="text-sm text-zinc-400">AI Financial Analyst</span>
          </div>
        )}

        {activeConvId ? (
          <ChatInterface key={activeConvId} conversationId={activeConvId} />
        ) : (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-4">
            <span className="text-5xl">📊</span>
            <div>
              <p className="text-zinc-300 font-medium">Select or start a conversation</p>
              <p className="text-zinc-500 text-sm mt-1">
                Use the sidebar to open a past chat or create a new one.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
