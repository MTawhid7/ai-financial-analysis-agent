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

  // Build a descriptive message when a file is parsed
  const handleFileParsed = (summary: Record<string, unknown>, filename: string) => {
    const ext = filename.split(".").pop()?.toLowerCase();
    let description = `📎 **${filename}** uploaded.`;

    if (ext === "csv") {
      const shape = summary.shape as { rows: number; columns: number } | undefined;
      const cols = (summary.columns as string[] | undefined)?.join(", ");
      if (shape) description += ` ${shape.rows} rows × ${shape.columns} columns.`;
      if (cols) description += ` Columns: ${cols}.`;
      if ((summary.formula_cells_removed as number) > 0)
        description += ` ⚠️ ${summary.formula_cells_removed} formula cell(s) were sanitised.`;
    } else if (ext === "pdf") {
      const pages = summary.pages as number | undefined;
      if (pages) description += ` ${pages} page(s).`;
      const s = summary.summary as string | undefined;
      if (s) description += `\n\n${s}`;
    }

    description += "\n\nWhat would you like to do with this file?";

    setMessages((prev) => [
      ...prev,
      {
        id: `file-${Date.now()}`,
        role: "assistant",
        content: description,
      },
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
      (detail) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `⚠️ Error: ${detail}`, isStreaming: false }
              : m
          )
        );
        setIsStreaming(false);
      }
    );
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
            <span className="text-5xl">📊</span>
            <div>
              <p className="text-zinc-300 font-medium">AI Financial Analyst</p>
              <p className="text-zinc-500 text-sm mt-1 max-w-xs">
                Ask me to analyse stocks, upload a CSV or PDF, or recall past research.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 justify-center mt-2">
              {["Analyse AAPL", "What is P/E ratio?", "Compare MSFT and GOOGL"].map((s) => (
                <button
                  key={s}
                  onClick={() => handleSend(s)}
                  className="text-xs px-3 py-1.5 rounded-full border border-zinc-700 text-zinc-400 hover:border-violet-500 hover:text-violet-400 transition-colors"
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

      {/* Input area with file upload */}
      <div className="border-t border-zinc-800">
        <div className="flex items-center gap-2 px-4 pt-2">
          <FileUploadZone onParsed={handleFileParsed} disabled={isStreaming} />
        </div>
        <MessageInput onSend={handleSend} disabled={isStreaming} />
      </div>
    </div>
  );
}
