import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getConversation, updateConversationTitle, uploadFile, type MessageOut } from "../../lib/api";
import {
  useStreamingChat,
  type ChartDescriptor,
  type CompleteEvent,
  type StepEvent,
} from "../../hooks/useStreamingChat";
import { AssistantBubble, TypingIndicator, UserBubble } from "./ChatBubble";
import { MessageInput } from "./MessageInput";

interface Props {
  conversationId: string;
  initialMessage?: string;
  onInitialMessageConsumed?: () => void;
}

interface DisplayMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  stepEvents?: StepEvent[];
  charts?: ChartDescriptor[];
  reportId?: string | null;
}

interface PendingAttachment {
  filename: string;
  description: string;
}

export function ChatInterface({ conversationId, initialMessage, onInitialMessageConsumed }: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [pendingAttachment, setPendingAttachment] = useState<PendingAttachment | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { streamMessage } = useStreamingChat();
  const qc = useQueryClient();
  const initialSentRef = useRef(false);
  // Guard: only load from DB once per component instance. Without this, getConversation
  // resolving for a brand-new conversation wipes locally-added streaming messages.
  const dbLoadedRef = useRef(false);

  const { data: detail } = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => getConversation(conversationId),
    enabled: !!conversationId,
  });

  useEffect(() => {
    if (detail && !dbLoadedRef.current) {
      dbLoadedRef.current = true;
      setMessages(
        detail.messages.map((m: MessageOut, i: number) => ({
          id: `${conversationId}-${i}`,
          role: m.role,
          content: m.content,
        }))
      );
    }
  }, [detail, conversationId]);

  // Fire initialMessage on first mount (from landing input)
  useEffect(() => {
    if (initialMessage && !initialSentRef.current) {
      initialSentRef.current = true;
      handleSend(initialMessage);
      onInitialMessageConsumed?.();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialMessage]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Build description string from file summary (used in pendingAttachment)
  const buildFileDescription = (summary: Record<string, unknown>, filename: string): string => {
    const ext = filename.split(".").pop()?.toLowerCase() ?? "";
    let desc = `File: ${filename}`;

    if (ext === "csv") {
      const shape = summary.shape as { rows: number; columns: number } | undefined;
      const cols = (summary.columns as string[] | undefined)?.slice(0, 8).join(", ");
      if (shape) desc += ` (${shape.rows.toLocaleString()} rows × ${shape.columns} columns)`;
      if (cols) desc += `. Columns: ${cols}${(summary.columns as string[])?.length > 8 ? "…" : ""}`;
      const removed = summary.formula_cells_removed as number | undefined;
      if (removed && removed > 0) desc += `. ⚠️ ${removed} formula injection(s) removed`;
    } else if (ext === "pdf") {
      const pages = summary.pages as number | undefined;
      if (pages) desc += ` (${pages} pages)`;
      const s = summary.summary as string | undefined;
      if (s) desc += `.\n\nSummary: ${s}`;
    } else if (["xlsx", "xls"].includes(ext)) {
      const sheets = summary.sheets as Record<string, unknown> | undefined;
      const sheetNames = sheets ? Object.keys(sheets).join(", ") : "";
      if (sheetNames) desc += `. Sheets: ${sheetNames}`;
    } else if (ext === "docx") {
      const s = summary.summary as string | undefined;
      if (s) desc += `.\n\nSummary: ${s}`;
    } else if (["txt", "md"].includes(ext)) {
      const chars = summary.char_count as number | undefined;
      if (chars) desc += ` (${chars.toLocaleString()} characters)`;
      const excerpt = summary.excerpt as string | undefined;
      if (excerpt) desc += `.\n\nExcerpt: ${excerpt}`;
    } else if (ext === "json") {
      const keys = summary.top_level_keys as string[] | undefined;
      if (keys) desc += `. Keys: ${keys.join(", ")}`;
    }
    return desc;
  };

  const handleFileUpload = async (file: File) => {
    try {
      const summary = await uploadFile(file);
      if (summary.error) {
        setPendingAttachment({ filename: file.name, description: `Upload error: ${summary.error}` });
        return;
      }
      const description = buildFileDescription(summary, file.name);
      setPendingAttachment({ filename: file.name, description });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setPendingAttachment({ filename: file.name, description: `Upload error: ${msg}` });
    }
  };

  const handleSend = async (text: string) => {
    if (isStreaming) return;

    // If there's a pending attachment, prepend its context to the message
    const isFirstMsg = messages.filter((m) => m.role === "user").length === 0;
    const fullText = pendingAttachment
      ? `[Attached file: ${pendingAttachment.filename}]\n${pendingAttachment.description}\n\nUser question: ${text}`
      : text;

    setPendingAttachment(null);

    const userMsg: DisplayMessage = { id: `user-${Date.now()}`, role: "user", content: text };
    const assistantId = `assistant-${Date.now()}`;
    const assistantMsg: DisplayMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
      stepEvents: [],
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    await streamMessage(
      conversationId,
      fullText,
      (step) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, stepEvents: [...(m.stepEvents ?? []), step] }
              : m
          )
        );
      },
      (event: CompleteEvent) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: event.response,
                  isStreaming: false,
                  charts: event.charts ?? [],
                  reportId: event.report_id ?? null,
                }
              : m
          )
        );
        setIsStreaming(false);

        // Auto-title: after first exchange, set title from user's first message
        if (isFirstMsg) {
          const title = text.slice(0, 60).trim();
          updateConversationTitle(conversationId, title)
            .then(() => qc.invalidateQueries({ queryKey: ["conversations"] }))
            .catch(() => {});
        }
      },
      (errDetail) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `⚠️ Error: ${errDetail}`, isStreaming: false }
              : m
          )
        );
        setIsStreaming(false);
      }
    );
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages scroll area */}
      <div className="flex-1 overflow-y-auto px-6 py-8 space-y-8">
        {messages.length === 0 && !initialMessage && (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
            <div className="w-12 h-12 rounded-2xl bg-zinc-800 border border-zinc-700 flex items-center justify-center">
              <svg className="w-6 h-6 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
            </div>
            <div className="space-y-2">
              <h2 className="text-2xl font-semibold text-zinc-100 tracking-tight">
                What would you like to analyse?
              </h2>
              <p className="text-zinc-500 text-sm max-w-sm leading-relaxed">
                Ask about stocks, compare companies, upload financial documents,
                or recall past research.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg) =>
          msg.role === "user" ? (
            <UserBubble key={msg.id} content={msg.content} />
          ) : (
            <AssistantBubble
              key={msg.id}
              content={msg.content}
              isStreaming={msg.isStreaming}
              stepEvents={msg.stepEvents}
              charts={msg.charts}
              reportId={msg.reportId}
            />
          )
        )}

        {isStreaming && messages[messages.length - 1]?.role === "user" && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-zinc-800/60">
        <MessageInput
          onSend={handleSend}
          disabled={isStreaming}
          pendingAttachment={pendingAttachment}
          onAttach={handleFileUpload}
          onClearAttachment={() => setPendingAttachment(null)}
        />
      </div>
    </div>
  );
}
