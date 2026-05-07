import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getConversation, type MessageOut } from "../../lib/api";
import { useStreamingChat, type StepEvent } from "../../hooks/useStreamingChat";
import { AssistantBubble, TypingIndicator, UserBubble } from "./ChatBubble";
import { MessageInput } from "./MessageInput";

interface Props {
  conversationId: string;
}

interface DisplayMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  // For in-flight assistant messages:
  isStreaming?: boolean;
  stepEvents?: StepEvent[];
}

export function ChatInterface({ conversationId }: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { streamMessage } = useStreamingChat();

  // Load existing messages when conversation changes
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

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (text: string) => {
    if (isStreaming) return;

    const userMsg: DisplayMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
    };
    const assistantMsgId = `assistant-${Date.now()}`;
    const assistantMsg: DisplayMessage = {
      id: assistantMsgId,
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
            m.id === assistantMsgId
              ? { ...m, stepEvents: [...(m.stepEvents ?? []), step] }
              : m
          )
        );
      },
      (response) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: response, isStreaming: false, stepEvents: m.stepEvents }
              : m
          )
        );
        setIsStreaming(false);
      },
      (detail) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  content: `⚠️ Error: ${detail}`,
                  isStreaming: false,
                }
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
                Ask me to analyse stocks, explain financial concepts, or recall previous research.
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
            />
          )
        )}

        {isStreaming && messages[messages.length - 1]?.role === "user" && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      <MessageInput onSend={handleSend} disabled={isStreaming} />
    </div>
  );
}
