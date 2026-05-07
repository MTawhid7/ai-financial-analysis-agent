import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { StepEvent } from "../../hooks/useStreamingChat";

interface UserBubbleProps {
  content: string;
}

interface AssistantBubbleProps {
  content: string;
  isStreaming?: boolean;
  stepEvents?: StepEvent[];
}

export function UserBubble({ content }: UserBubbleProps) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-violet-600 px-4 py-2.5 text-sm text-white shadow-sm whitespace-pre-wrap break-words">
        {content}
      </div>
    </div>
  );
}

export function AssistantBubble({ content, isStreaming, stepEvents }: AssistantBubbleProps) {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-sm mt-0.5">
        📊
      </div>
      <div className="flex-1 min-w-0 space-y-2">
        {/* Live tool step indicators shown during streaming */}
        {(isStreaming || (stepEvents && stepEvents.length > 0)) && (
          <div className="space-y-1 font-mono">
            {stepEvents?.map((evt, i) => (
              <div key={i} className="flex items-center gap-1.5 text-xs text-zinc-500">
                <span className={evt.ok !== false ? "text-emerald-500" : "text-red-500"}>
                  {evt.ok !== false ? "✓" : "✗"}
                </span>
                <span>
                  [{evt.step}] {evt.agent} → {evt.tool}
                  {evt.cache_hit ? " (cached)" : ""}
                </span>
              </div>
            ))}
            {isStreaming && (
              <div className="flex items-center gap-1.5 text-xs text-zinc-600">
                <span className="animate-pulse">●</span>
                <span>Processing…</span>
              </div>
            )}
          </div>
        )}

        {/* Rendered markdown response */}
        {content && (
          <div className="prose prose-sm prose-invert max-w-none text-zinc-200">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-sm">
        📊
      </div>
      <div className="flex items-center gap-1 py-2">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
    </div>
  );
}
