import { type KeyboardEvent, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createConversation,
  deleteConversation,
  listConversations,
  updateConversationTitle,
  type ConversationSummary,
} from "../../lib/api";

interface Props {
  activeId: string | null;
  onSelect: (id: string) => void;
}

function timeLabel(ts: number): string {
  const delta = Date.now() / 1000 - ts;
  if (delta < 86400) return "Today";
  if (delta < 172800) return "Yesterday";
  return new Date(ts * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function ConversationItem({
  conv,
  isActive,
  onSelect,
  onDelete,
  onRename,
}: {
  conv: ConversationSummary;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onRename: (title: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(conv.title);
  const inputRef = useRef<HTMLInputElement>(null);

  const startEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDraft(conv.title);
    setEditing(true);
    setTimeout(() => inputRef.current?.select(), 0);
  };

  const commitEdit = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== conv.title) {
      onRename(trimmed);
    }
    setEditing(false);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") { e.preventDefault(); commitEdit(); }
    if (e.key === "Escape") { setEditing(false); }
  };

  if (editing) {
    return (
      <div className="flex items-center rounded-lg px-2 py-1.5 bg-zinc-800">
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={commitEdit}
          autoFocus
          className="flex-1 text-xs text-zinc-200 bg-transparent outline-none border-b border-zinc-600 pb-0.5"
        />
      </div>
    );
  }

  return (
    <div
      className={`group flex items-center rounded-lg px-2 py-1.5 cursor-pointer transition-colors ${
        isActive
          ? "bg-zinc-800 text-zinc-100"
          : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
      }`}
      onClick={onSelect}
    >
      <span className="flex-1 text-xs truncate">{conv.title}</span>
      <span className="text-[10px] text-zinc-600 mr-1 flex-shrink-0 group-hover:hidden">
        {timeLabel(conv.updated_at)}
      </span>

      {/* Action icons — rename + delete */}
      <div className="hidden group-hover:flex items-center gap-0.5 flex-shrink-0">
        {/* Rename */}
        <button
          onClick={startEdit}
          className="p-0.5 rounded hover:text-zinc-200 transition-colors"
          aria-label="Rename conversation"
          title="Rename"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
        </button>
        {/* Delete */}
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="p-0.5 rounded hover:text-red-400 transition-colors"
          aria-label="Delete conversation"
          title="Delete"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}

export function ConversationList({ activeId, onSelect }: Props) {
  const qc = useQueryClient();

  const { data: conversations = [] } = useQuery({
    queryKey: ["conversations"],
    queryFn: listConversations,
  });

  const createMut = useMutation({
    mutationFn: () => createConversation("New conversation"),
    onSuccess: (conv) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      onSelect(conv.id);
    },
  });

  const deleteMut = useMutation({
    mutationFn: deleteConversation,
    onSuccess: (_, deletedId) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      if (activeId === deletedId) onSelect("");
    },
  });

  const renameMut = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) =>
      updateConversationTitle(id, title),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });

  return (
    <div className="flex flex-col gap-1">
      <button
        onClick={() => createMut.mutate()}
        disabled={createMut.isPending}
        className="flex items-center gap-2 w-full rounded-lg px-3 py-2 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
        New conversation
      </button>

      <div className="mt-2 space-y-0.5">
        {conversations.map((conv: ConversationSummary) => (
          <ConversationItem
            key={conv.id}
            conv={conv}
            isActive={activeId === conv.id}
            onSelect={() => onSelect(conv.id)}
            onDelete={() => deleteMut.mutate(conv.id)}
            onRename={(title) => renameMut.mutate({ id: conv.id, title })}
          />
        ))}

        {conversations.length === 0 && (
          <p className="text-xs text-zinc-600 px-2 py-2">No conversations yet.</p>
        )}
      </div>
    </div>
  );
}
