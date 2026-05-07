import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createConversation,
  deleteConversation,
  listConversations,
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
          <div
            key={conv.id}
            className={`group flex items-center rounded-lg px-2 py-1.5 cursor-pointer transition-colors ${
              activeId === conv.id
                ? "bg-zinc-800 text-zinc-100"
                : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
            }`}
            onClick={() => onSelect(conv.id)}
          >
            <span className="flex-1 text-xs truncate">{conv.title}</span>
            <span className="text-[10px] text-zinc-600 mr-1.5 flex-shrink-0">
              {timeLabel(conv.updated_at)}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                deleteMut.mutate(conv.id);
              }}
              className="flex-shrink-0 opacity-0 group-hover:opacity-100 p-0.5 rounded hover:text-red-400 transition-all"
              aria-label="Delete conversation"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        ))}

        {conversations.length === 0 && (
          <p className="text-xs text-zinc-600 px-2 py-2">No conversations yet.</p>
        )}
      </div>
    </div>
  );
}
