import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getConversation, type MessageOut } from "../../lib/api";
import {
  useStreamingChat,
  type ChartDescriptor,
  type CompleteEvent,
  type StepEvent,
} from "../../hooks/useStreamingChat";
import { AssistantBubble, TypingIndicator, UserBubble } from "./ChatBubble";
import { MessageInput } from "./MessageInput";
import { FileUploadZone } from "./FileUploadZone";

interface Props {
  conversationId: string;
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

const SUGGESTION_CHIPS = [
  "Analyse AAPL",
  "Compare AAPL vs MSFT",
  "What is a P/E ratio?",
  "What did we find about NVDA?",
];

export function ChatInterface({ conversationId }: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { streamMessage } = useStreamingChat();

  const { data: detail } = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => getConversation(conversationId),
    enabled: !!conversationId,
  });

  useEffect(() => {
    if (detail) {
      setMessages(
        detail.messages.map((m: MessageOut, i: number) => ({
          id: `${conversationId}-${i}`,
          role: m.role,
          content: m.content,
        }))
      );
    }
  }, [detail, conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Build descriptive message for uploaded files
  const handleFileParsed = (summary: Record<string, unknown>, filename: string) => {
    const ext = filename.split(".").pop()?.toLowerCase() ?? "";
    let description = `📎 **${filename}** uploaded.`;

    if (ext === "csv") {
      const shape = summary.shape as { rows: number; columns: number } | undefined;
      const cols = (summary.columns as string[] | undefined)?.slice(0, 8).join(", ");
      if (shape) description += ` ${shape.rows.toLocaleString()} rows × ${shape.columns} columns.`;
      if (cols) description += ` Columns: ${cols}${(summary.columns as string[])?.length > 8 ? "…" : ""}.`;
      const removed = summary.formula_cells_removed as number | undefined;
      if (removed && removed > 0)
        description += ` ⚠️ ${removed} formula injection(s) removed.`;
    } else if (ext === "pdf") {
      const pages = summary.pages as number | undefined;
      if (pages) description += ` ${pages} page(s).`;
      const s = summary.summary as string | undefined;
      if (s) description += `\n\n${s}`;
    } else if (["xlsx", "xls"].includes(ext)) {
      const sheets = summary.sheets as Record<string, unknown> | undefined;
      const sheetNames = sheets ? Object.keys(sheets).join(", ") : "";
      if (sheetNames) description += ` Sheets: ${sheetNames}.`;
    } else if (ext === "docx") {
      const s = summary.summary as string | undefined;
      if (s) description += `\n\n${s}`;
    } else if (["txt", "md"].includes(ext)) {
      const chars = summary.char_count as number | undefined;
      if (chars) description += ` ${chars.toLocaleString()} characters.`;
      const excerpt = summary.excerpt as string | undefined;
      if (excerpt) description += `\n\n${excerpt}`;
    } else if (ext === "json") {
      const keys = summary.top_level_keys as string[] | undefined;
      if (keys) description += ` Keys: ${keys.join(", ")}.`;
    }
    description += "\n\nWhat would you like to do with this file?";

    setMessages((prev) => [
      ...prev,
      { id: `file-${Date.now()}`, role: "assistant", content: description },
    ]);
  };

  const handleSend = async (text: string) => {
    if (isStreaming) return;

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
      text,
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
        {messages.length === 0 && (
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
            <div className="flex flex-wrap gap-2 justify-center max-w-md">
              {SUGGESTION_CHIPS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSend(s)}
                  className="text-sm px-4 py-2 rounded-full border border-zinc-700 text-zinc-400 hover:border-violet-500 hover:text-violet-400 hover:bg-violet-950/30 transition-all duration-150"
                >
                  {s}
                </button>
              ))}
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
        <div className="px-4 pt-2 pb-1">
          <FileUploadZone onParsed={handleFileParsed} disabled={isStreaming} />
        </div>
        <MessageInput onSend={handleSend} disabled={isStreaming} />
      </div>
    </div>
  );
}
